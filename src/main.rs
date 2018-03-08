extern crate cpal;
extern crate wavefile;

mod mixer;

use std::process::exit;
use wavefile::WaveFile;

fn main() {
    let device = cpal::default_output_device().expect("Failed to get default output device");
    let mut format = device
        .default_output_format()
        .expect("Failed to get default output format");
    format.data_type = cpal::SampleFormat::I16;
    let event_loop = cpal::EventLoop::new();
    let stream_id = event_loop.build_output_stream(&device, &format).unwrap();
    event_loop.play_stream(stream_id.clone());

    let wav = match WaveFile::open("src/loop.wav") {
        Ok(w) => w,
        Err(e) => panic!("Oh no: {}", e),
    };

    let mut wav_iter = wav.iter();

    event_loop.run(move |_, data| match data {
        cpal::StreamData::Output {
            buffer: cpal::UnknownTypeOutputBuffer::U16(mut buffer),
        } => for sample in buffer.chunks_mut(format.channels as usize) {
            let mut count = 0;
            let value = match wav_iter.next() {
                Some(v) => v,
                None => exit(0),
            };
            for out in sample.iter_mut() {
                *out = value[count] as u16;
            }
        },
        cpal::StreamData::Output {
            buffer: cpal::UnknownTypeOutputBuffer::I16(mut buffer),
        } => for sample in buffer.chunks_mut(format.channels as usize) {
            let mut count = 0;
            let value = match wav_iter.next() {
                Some(v) => v,
                None => exit(0),
            };
            for out in sample.iter_mut() {
                *out = value[count] as i16;
            }
        },
        cpal::StreamData::Output {
            buffer: cpal::UnknownTypeOutputBuffer::F32(mut buffer),
        } => for sample in buffer.chunks_mut(format.channels as usize) {
            let mut count = 0;
            let value = match wav_iter.next() {
                Some(v) => v,
                None => exit(0),
            };
            for out in sample.iter_mut() {
                *out = value[count] as f32;
            }
        },
        _ => (),
    });
}
