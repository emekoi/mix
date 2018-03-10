#![crate_type = "lib"]
#![allow(unused)]

use std::cell::RefCell;
use std::cell::RefMut;
use std::any::Any;
use std::rc::Rc;

// https://github.com/rxi/cmixer/blob/master/src/cmixer.h
// https://github.com/rxi/cmixer/blob/master/src/cmixer.c
// https://ricardomartins.cc/2016/06/08/interior-mutability

macro_rules! FX_FROM_FLOAT {
  ($f:expr) => {
      ($f * FX_UNIT as f32) as isize
  };
}

macro_rules! FX_LERP {
  ($a:expr, $b:expr, $p:expr) => {
      ($a) + (((($b) - ($a)) * ($p)) >> FX_BITS)
  };
}

const FX_BITS: usize = 12;
const FX_UNIT: usize = 1 << FX_BITS;
const FX_MASK: usize = FX_UNIT - 1;
const BUFFER_SIZE: usize = 512;
const BUFFER_MASK: usize = BUFFER_SIZE - 1;

#[derive(PartialEq, Copy, Clone)]
pub enum State {
    STOPPED,
    PLAYING,
    PAUSED,
}

impl Default for State {
    fn default() -> Self {
        State::STOPPED
    }
}

#[derive(PartialEq, Copy, Clone)]
pub enum EventType {
    NULL,
    LOCK,
    UNLOCK,
    DESTROY,
    SAMPLES,
    REWIND,
}

impl Default for EventType {
    fn default() -> Self {
        EventType::NULL
    }
}

pub struct Event<T: Clone> {
    kind: EventType,
    udata: Option<T>,
    msg: String,
    buffer: Vec<i16>,
    length: isize,
}

pub type EventHandler<T> = fn(&Event<T>);

pub struct SourceInfo<T: Clone> {
    handler: EventHandler<T>,
    udata: Option<T>,
    samplerate: isize,
    length: isize,
}

#[derive(Clone)]
struct _Source<T: Clone> {
    buffer: [i16; BUFFER_SIZE], /* Internal buffer with raw stereo PCM */
    handler: EventHandler<T>,   /* Event handler */
    udata: Option<T>,           /* Stream's udata (from SourceInfo) */
    samplerate: isize,          /* Stream's native samplerate */
    length: isize,              /* Stream's length in frames */
    end: isize,                 /* End index for the current play-through */
    state: State,               /* Current state (playing|paused|stopped) */
    position: isize,            /* Current playhead position (fixed point) */
    lgain: isize,               /* Left gain (fixed point) */
    rgain: isize,               /* Right gain (fixed point) */
    rate: isize,                /* Playback rate (fixed point) */
    nextfill: isize,            /* Next frame idx where the buffer needs to be filled */
    is_loop: bool,              /* Whether the source will loop when `end` is reached */
    rewind: bool,               /* Whether the source will rewind before playing */
    active: bool,               /* Whether the source is part of `sources` list */
    gain: f32,                  /* Gain set by `set_gain()` */
    pan: f32,                   /* Pan set by `set_pan()` */
    mixer: MixerRef<T>,         /* The mixer that created this source */
    id: usize,                  /* A unique(ish) identifier for each source */
}

type SourceRef<T> = Rc<RefCell<_Source<T>>>;

#[derive(Clone)]
struct _Mixer<T: Clone> {
    last_error: String,            /* Last error message */
    lock_handler: EventHandler<T>, /* Event handler for lock/unlock events */
    sources: Vec<SourceRef<T>>,    /* List of active (playing) sources */
    buffer: [i32; BUFFER_SIZE],    /* Internal master buffer */
    samplerate: isize,             /* Master samplerate */
    gain: isize,                   /* Master gain (fixed point) */
}

type MixerRef<T> = Rc<RefCell<_Mixer<T>>>;

pub struct Mixer<T: Clone>(MixerRef<T>);
pub struct Source<T: Clone>(SourceRef<T>);

impl<T: Clone> Event<T> {
    pub fn new() -> Self {
        Event {
            kind: EventType::NULL,
            udata: None,
            msg: String::from(""),
            buffer: vec![0i16; 0],
            length: 0,
        }
    }
}

impl<T: Clone> Default for Event<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone> Source<T> {
    pub fn get_length(&self) -> f32 {
        let src = self.0.borrow();
        src.length as f32 / src.samplerate as f32
    }

    pub fn get_position(&self) -> f32 {
        let src = self.0.borrow();
        ((src.position >> FX_BITS) % src.length) as f32 / src.samplerate as f32
    }

    pub fn get_state(&self) -> State {
        self.0.borrow().state
    }

    fn recalc_source_gains(&self) {
        let mut src = self.0.borrow_mut();
        let pan = src.pan;
        let l = src.gain * if pan <= 0.0 { 1.0 } else { 1.0 - pan };
        let r = src.gain * if pan >= 0.0 { 1.0 } else { 1.0 + pan };
        src.lgain = FX_FROM_FLOAT!(l);
        src.rgain = FX_FROM_FLOAT!(r);
    }

    pub fn set_gain(&self, gain: f32) {
        let mut src = self.0.borrow_mut();
        src.gain = gain;
        self.recalc_source_gains();
    }

    pub fn set_pan(&self, pan: f32) {
        let mut src = self.0.borrow_mut();
        src.pan = pan.max(-1.0).min(1.0);
        self.recalc_source_gains();
    }

    pub fn set_pitch(&self, pitch: f32) {
        let mut src = self.0.borrow_mut();
        let rate = if pitch > 0.0 {
            (src.samplerate / src.mixer.borrow().samplerate) as f32 * pitch
        } else {
            0.001
        };
        src.rate = FX_FROM_FLOAT!(rate);
    }

    pub fn set_loop(&self, is_loop: bool) {
        let mut src = self.0.borrow_mut();
        src.is_loop = is_loop;
    }

    pub fn play(&self) {
        let mut src = self.0.borrow_mut();
        src.mixer.borrow().lock();
        src.state = State::PLAYING;
        if !src.active {
            src.active = true;
            src.mixer.borrow_mut().sources.push(self.0.clone());
        }
        src.mixer.borrow().unlock();
    }

    pub fn pause(&mut self) {
        let mut src = self.0.borrow_mut();
        src.state = State::PAUSED;
    }

    pub fn stop(&mut self) {
        let mut src = self.0.borrow_mut();
        src.state = State::STOPPED;
        src.rewind = true;
    }
}

impl<T: Clone> Drop for Source<T> {
    fn drop(&mut self) {
        let mut src = self.0.borrow_mut();
        let mut e = Event::new();
        src.mixer.borrow().lock();
        if src.active {
            src.mixer
                .borrow_mut()
                .sources
                .retain(|s| s.borrow().id != src.id);
        }
        src.mixer.borrow().unlock();
        e.kind = EventType::DESTROY;
        e.udata = src.udata.clone();
        (src.handler)(&e);
    }
}

impl<T: Clone> _Mixer<T> {
    fn lock(&self) {
        let mut e = Event::new();
        e.kind = EventType::LOCK;
        (self.lock_handler)(&e);
    }

    fn unlock(&self) {
        let mut e = Event::new();
        e.kind = EventType::UNLOCK;
        (self.lock_handler)(&e);
    }
}

impl<T: Clone> Mixer<T> {
    pub fn init(samplerate: isize) -> Self {
        Mixer(Rc::new(RefCell::new(_Mixer {
            last_error: String::from(""),
            samplerate,
            lock_handler: |e| {},
            sources: vec![],
            gain: FX_UNIT as isize,
            buffer: [0; BUFFER_SIZE],
        })))
    }

    pub fn get_error(&self) -> String {
        let mut mixer = self.0.borrow_mut();
        let res = mixer.last_error.clone();
        mixer.last_error = String::from("");
        res
    }

    fn error(&self, msg: String) -> String {
        self.0.borrow_mut().last_error = msg.clone();
        msg
    }

    pub fn set_lock(&self, e: EventHandler<T>) {
        self.0.borrow_mut().lock_handler = e;
    }

    pub fn set_master_gain(&self, gain: f32) {
        self.0.borrow_mut().gain = FX_FROM_FLOAT!(gain);
    }

    fn rewind_source(src: &mut RefMut<_Source<T>>) {
        let mut e = Event::new();
        e.kind = EventType::REWIND;
        e.udata = src.udata.clone();
        (src.handler)(&e);
        src.position = 0;
        src.rewind = false;
        src.end = src.length;
        src.nextfill = 0;
    }

    fn fill_source_buffer(src: &RefMut<_Source<T>>, offset: usize, length: usize) {
        let mut e = Event::new();
        e.kind = EventType::SAMPLES;
        e.udata = src.udata.clone();
        if offset > src.buffer.len() {
            panic!("offset too large");
        }
        // for i in 0..src.buffer[offset..].len() {
        //     e.buffer[i] = src.buffer[offset + i];
        // }
        e.buffer[..src.buffer[offset..].len()]
            .clone_from_slice(&src.buffer[offset..(src.buffer[offset..].len() + offset)]);
        e.length = length as isize;
        (src.handler)(&e);
    }

    fn process_source(&self, src: &SourceRef<T>, len: isize) {
        let mut dst = &mut self.0.borrow_mut().buffer[..];
        let mut src = src.borrow_mut();

        /* Do rewind if flag is set */
        if src.rewind {
            Mixer::rewind_source(&mut src);
        }

        /* Don't process if not playing */
        if src.state != State::PLAYING {
            return;
        }

        let (mut n, mut a, mut b, mut p): (isize, isize, isize, isize);
        let mut frame = 0;
        let mut count = 0;
        let mut len = len;
        let mut offset = 0;

        /* Process audio */
        while len > 0 {
            /* Get current position frame */
            frame = src.position >> FX_BITS;

            /* Fill buffer if required */
            if frame + 3 >= src.nextfill {
                Mixer::fill_source_buffer(
                    &src,
                    (src.nextfill * 2) as usize & BUFFER_MASK,
                    BUFFER_SIZE / 2,
                );
                src.nextfill += (BUFFER_SIZE / 4) as isize;
            }

            /* Handle reaching the end of the playthrough */
            if frame >= src.end {
                /* As streams continiously fill the raw buffer in a loop we simply
                 ** increment the end idx by one length and continue reading from it for
                 ** another play-through */
                src.end = frame + src.length;
                /* Set state and stop processing if we're not set to loop */
                if !src.is_loop {
                    src.state = State::STOPPED;
                    break;
                }
            }

            /* Work out how many frames we should process in the loop */
            n = (src.nextfill - 2).min(src.end) - frame;
            count = (n << FX_BITS) / src.rate;
            count = (count).max(1);
            count = (count).min(len / 2);
            len -= count * 2;

            /* Add audio to master buffer */
            if src.rate == FX_UNIT as isize {
                /* Add audio to buffer -- basic */
                n = frame * 2;
                for i in 0..count {
                    dst[offset] += i32::from(
                        src.buffer[(n) as usize & BUFFER_MASK] * src.lgain as i16,
                    ) >> FX_BITS;
                    dst[offset + 1] += i32::from(
                        src.buffer[(n + 1) as usize & BUFFER_MASK] * src.rgain as i16,
                    ) >> FX_BITS;
                    n += 2;
                    offset += 2;
                }
                src.position += count * FX_UNIT as isize;
            } else {
                /* Add audio to buffer -- interpolated */
                for i in 0..count {
                    n = (src.position >> FX_BITS) * 2;
                    p = src.position & FX_MASK as isize;
                    a = src.buffer[(n) as usize & BUFFER_MASK] as isize;
                    b = src.buffer[(n + 2) as usize & BUFFER_MASK] as isize;
                    dst[offset] += (FX_LERP!(a, b, p) * src.lgain) as i32 >> FX_BITS;
                    n += 1;
                    a = src.buffer[(n) as usize & BUFFER_MASK] as isize;
                    b = src.buffer[(n + 2) as usize & BUFFER_MASK] as isize;
                    dst[offset + 1] += (FX_LERP!(a, b, p) * src.rgain) as i32 >> FX_BITS;
                    src.position += src.rate;
                    offset += 2;
                }
            }
        }
    }

    pub fn process(&self, dst: &mut [i16], mut len: usize) {
        let mut mixer = self.0.borrow_mut();
        let mut dst = dst;
        /* Process in chunks of BUFFER_SIZE if `len` is larger than BUFFER_SIZE */
        let mut offset = 0;
        while len > BUFFER_SIZE {
            self.process(&mut dst[(BUFFER_SIZE * offset)..], BUFFER_SIZE);
            len -= BUFFER_SIZE;
            offset += 1;
        }

        /* Zeroset internal buffer */
        mixer.buffer = [0; BUFFER_SIZE];

        /* Process active sources */
        self.0.borrow().lock();
        for s in &mixer.sources {
            self.process_source(s, len as isize);
            let mut s = s.borrow_mut();
            /* Remove source from list if it is no longer playing */
            if s.state != State::PLAYING {
                s.active = false;
            }
        }
        self.0.borrow().unlock();

        /* Copy internal buffer to destination and clip */
        // for i in 0..len {
        for (i, d) in dst.iter_mut().enumerate().take(len) {
            let x = (mixer.buffer[i] * mixer.gain as i32) >> FX_BITS;
            *d = x.max(-32_768).min(32_767) as i16;
        }
    }

    pub fn new_source(&self, info: SourceInfo<T>) -> Source<T> {
        let mut src = Source(Rc::new(RefCell::new(_Source {
            buffer: [0; BUFFER_SIZE],
            handler: info.handler,
            udata: info.udata,
            samplerate: info.samplerate,
            length: info.length,
            end: 0,
            state: State::STOPPED,
            position: 0,
            lgain: 0,
            rgain: 0,
            rate: 0,
            nextfill: 0,
            is_loop: false,
            rewind: false,
            active: false,
            gain: 0.0,
            pan: 0.0,
            mixer: self.0.clone(),
            id: 0,
        })));
        src.set_gain(1.0);
        src.set_pan(0.0);
        src.set_pitch(1.0);
        src.set_loop(false);
        src.stop();
        src.0.borrow_mut().id = unsafe { ::std::mem::transmute(&src as *const Source<T>) };
        src
    }
}

/*
static const char* wav_init(cm_SourceInfo *info, void *data, int len, int ownsdata);

#ifdef CM_USE_STB_VORBIS
static const char* ogg_init(cm_SourceInfo *info, void *data, int len, int ownsdata);
#endif


static int check_header(void *data, int size, char *str, int offset) {
  int len = strlen(str);
  return (size >= offset + len) && !memcmp((char*) data + offset, str, len);
}


static cm_Source* new_source_from_mem(void *data, int size, int ownsdata) {
  const char *err;
  cm_SourceInfo info;

  if (check_header(data, size, "WAVE", 8)) {
    err = wav_init(&info, data, size, ownsdata);
    if (err) {
      return NULL;
    }
    return cm_new_source(&info);
  }

#ifdef CM_USE_STB_VORBIS
  if (check_header(data, size, "OggS", 0)) {
    err = ogg_init(&info, data, size, ownsdata);
    if (err) {
      return NULL;
    }
    return cm_new_source(&info);
  }
#endif

  error("unknown format or invalid data");
  return NULL;
}


static void* load_file(const char *filename, int *size) {
  FILE *fp;
  void *data;
  int n;

  fp = fopen(filename, "rb");
  if (!fp) {
    return NULL;
  }

  /* Get size */
  fseek(fp, 0, SEEK_END);
  *size = ftell(fp);
  rewind(fp);

  /* Malloc, read and return data */
  data = malloc(*size);
  if (!data) {
    fclose(fp);
    return NULL;
  }
  n = fread(data, 1, *size, fp);
  fclose(fp);
  if (n != *size) {
    free(data);
    return NULL;
  }

  return data;
}


cm_Source* cm_new_source_from_file(const char *filename) {
  int size;
  cm_Source *src;
  void *data;

  /* Load file into memory */
  data = load_file(filename, &size);
  if (!data) {
    error("could not load file");
    return NULL;
  }

  /* Try to load and return */
  src = new_source_from_mem(data, size, 1);
  if (!src) {
    free(data);
    return NULL;
  }

  return src;
}


cm_Source* cm_new_source_from_mem(void *data, int size) {
  return new_source_from_mem(data, size, 0);
}

/*============================================================================
** Wav stream
**============================================================================*/

typedef struct {
  void *data;
  int bitdepth;
  int samplerate;
  int channels;
  int length;
} Wav;

typedef struct {
  Wav wav;
  void *data;
  int idx;
} WavStream;


static char* find_subchunk(char *data, int len, char *id, int *size) {
  /* TODO : Error handling on malformed wav file */
  int idlen = strlen(id);
  char *p = data + 12;
next:
  *size = *((cm_UInt32*) (p + 4));
  if (memcmp(p, id, idlen)) {
    p += 8 + *size;
    if (p > data + len) return NULL;
    goto next;
  }
  return p + 8;
}


static const char* read_wav(Wav *w, void *data, int len) {
  int bitdepth, channels, samplerate, format;
  int sz;
  char *p = data;
  memset(w, 0, sizeof(*w));

  /* Check header */
  if (memcmp(p, "RIFF", 4) || memcmp(p + 8, "WAVE", 4)) {
    return error("bad wav header");
  }
  /* Find fmt subchunk */
  p = find_subchunk(data, len, "fmt", &sz);
  if (!p) {
    return error("no fmt subchunk");
  }

  /* Load fmt info */
  format      = *((cm_UInt16*) (p));
  channels    = *((cm_UInt16*) (p + 2));
  samplerate  = *((cm_UInt32*) (p + 4));
  bitdepth    = *((cm_UInt16*) (p + 14));
  if (format != 1) {
    return error("unsupported format");
  }
  if (channels == 0 || samplerate == 0 || bitdepth == 0) {
    return error("bad format");
  }

  /* Find data subchunk */
  p = find_subchunk(data, len, "data", &sz);
  if (!p) {
    return error("no data subchunk");
  }

  /* Init struct */
  w->data = (void*) p;
  w->samplerate = samplerate;
  w->channels = channels;
  w->length = (sz / (bitdepth / 8)) / channels;
  w->bitdepth = bitdepth;
  /* Done */
  return NULL;
}


#define WAV_PROCESS_LOOP(X) \
  while (n--) {             \
    X                       \
    dst += 2;               \
    s->idx++;               \
  }

static void wav_handler(cm_Event *e) {
  int x, n;
  cm_Int16 *dst;
  WavStream *s = e->udata;
  int len;

  switch (e->type) {

    case CM_EVENT_DESTROY:
      free(s->data);
      free(s);
      break;

    case CM_EVENT_SAMPLES:
      dst = e->buffer;
      len = e->length / 2;
fill:
      n = MIN(len, s->wav.length - s->idx);
      len -= n;
      if (s->wav.bitdepth == 16 && s->wav.channels == 1) {
        WAV_PROCESS_LOOP({
          dst[0] = dst[1] = ((cm_Int16*) s->wav.data)[s->idx];
        });
      } else if (s->wav.bitdepth == 16 && s->wav.channels == 2) {
        WAV_PROCESS_LOOP({
          x = s->idx * 2;
          dst[0] = ((cm_Int16*) s->wav.data)[x    ];
          dst[1] = ((cm_Int16*) s->wav.data)[x + 1];
        });
      } else if (s->wav.bitdepth == 8 && s->wav.channels == 1) {
        WAV_PROCESS_LOOP({
          dst[0] = dst[1] = (((cm_UInt8*) s->wav.data)[s->idx] - 128) << 8;
        });
      } else if (s->wav.bitdepth == 8 && s->wav.channels == 2) {
        WAV_PROCESS_LOOP({
          x = s->idx * 2;
          dst[0] = (((cm_UInt8*) s->wav.data)[x    ] - 128) << 8;
          dst[1] = (((cm_UInt8*) s->wav.data)[x + 1] - 128) << 8;
        });
      }
      /* Loop back and continue filling buffer if we didn't fill the buffer */
      if (len > 0) {
        s->idx = 0;
        goto fill;
      }
      break;

    case CM_EVENT_REWIND:
      s->idx = 0;
      break;
  }
}


static const char* wav_init(cm_SourceInfo *info, void *data, int len, int ownsdata) {
  WavStream *stream;
  Wav wav;

  const char *err = read_wav(&wav, data, len);
  if (err != NULL) {
    return err;
  }

  if (wav.channels > 2 || (wav.bitdepth != 16 && wav.bitdepth != 8)) {
    return error("unsupported wav format");
  }

  stream = calloc(1, sizeof(*stream));
  if (!stream) {
    return error("allocation failed");
  }
  stream->wav = wav;

  if (ownsdata) {
    stream->data = data;
  }
  stream->idx = 0;

  info->udata = stream;
  info->handler = wav_handler;
  info->samplerate = wav.samplerate;
  info->length = wav.length;

  /* Return NULL (no error) for success */
  return NULL;
}


/*============================================================================
** Ogg stream
**============================================================================*/

#ifdef CM_USE_STB_VORBIS

#define STB_VORBIS_HEADER_ONLY
#include "stb_vorbis.c"

typedef struct {
  stb_vorbis *ogg;
  void *data;
} OggStream;


static void ogg_handler(cm_Event *e) {
  int n, len;
  OggStream *s = e->udata;
  cm_Int16 *buf;

  switch (e->type) {

    case CM_EVENT_DESTROY:
      stb_vorbis_close(s->ogg);
      free(s->data);
      free(s);
      break;

    case CM_EVENT_SAMPLES:
      len = e->length;
      buf = e->buffer;
fill:
      n = stb_vorbis_get_samples_short_interleaved(s->ogg, 2, buf, len);
      n *= 2;
      /* rewind and fill remaining buffer if we reached the end of the ogg
      ** before filling it */
      if (len != n) {
        stb_vorbis_seek_start(s->ogg);
        buf += n;
        len -= n;
        goto fill;
      }
      break;

    case CM_EVENT_REWIND:
      stb_vorbis_seek_start(s->ogg);
      break;
  }
}


static const char* ogg_init(cm_SourceInfo *info, void *data, int len, int ownsdata) {
  OggStream *stream;
  stb_vorbis *ogg;
  stb_vorbis_info ogginfo;
  int err;

  ogg = stb_vorbis_open_memory(data, len, &err, NULL);
  if (!ogg) {
    return error("invalid ogg data");
  }

  stream = calloc(1, sizeof(*stream));
  if (!stream) {
    stb_vorbis_close(ogg);
    return error("allocation failed");
  }

  stream->ogg = ogg;
  if (ownsdata) {
    stream->data = data;
  }

  ogginfo = stb_vorbis_get_info(ogg);

  info->udata = stream;
  info->handler = ogg_handler;
  info->samplerate = ogginfo.sample_rate;
  info->length = stb_vorbis_stream_length_in_samples(ogg);

  /* Return NULL (no error) for success */
  return NULL;
}


#endif
*/

#[cfg(test)]
mod tests {
    #[test]
    fn test() {}
}
//
fn main() {}
