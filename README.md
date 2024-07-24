# whisperx-worker
RunPod Serverless worker for WhisperX

## License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.


## Input Options

| Parameter                    | Type    | Required | Default | Description                                                                 |
|------------------------------|---------|----------|---------|-----------------------------------------------------------------------------|
| `audio_file`                 | string  | Yes      | N/A     | URL to the audio file for transcription                                     |
| `language`                   | string  | No       | None    | ISO code of the language spoken in the audio. If not specified, language detection will be performed |
| `language_detection_min_prob`| float   | No       | 0       | Minimum probability for language detection                                  |
| `language_detection_max_tries`| int    | No       | 5       | Maximum number of tries for language detection                              |
| `initial_prompt`             | string  | No       | None    | Optional text to provide as a prompt for the first window                   |
| `batch_size`                 | int     | No       | 64      | Parallelization of input audio transcription                                |
| `temperature`                | float   | No       | 0       | Temperature to use for sampling                                             |
| `vad_onset`                  | float   | No       | 0.500   | Voice Activity Detection onset                                              |
| `vad_offset`                 | float   | No       | 0.363   | Voice Activity Detection offset                                             |
| `align_output`               | bool    | No       | False   | Aligns Whisper output to get accurate word-level timestamps                 |
| `diarization`                | bool    | No       | False   | Assign speaker ID labels                                                    |
| `huggingface_access_token`   | string  | No       | None    | HuggingFace token for diarization. Required if diarization is True          |
| `min_speakers`               | int     | No       | None    | Minimum number of speakers if diarization is activated                      |
| `max_speakers`               | int     | No       | None    | Maximum number of speakers if diarization is activated                      |
| `debug`                      | bool    | No       | False   | Print out compute/inference times and memory usage information              |

## Usage Example

Here's how to format your input when calling the service:

```json
{
  "input": {
    "audio_file": "https://example.com/audio/sample.mp3",
    "language": "en",
    "batch_size": 32,
    "temperature": 0.2,
    "align_output": true,
    "debug": true
  }
}
```

## Output

The service returns a JSON object structured as follows:

```json
{
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "Transcribed text segment 1"
    },
    {
      "start": 2.5,
      "end": 5.0,
      "text": "Transcribed text segment 2"
    }
    // ... more segments
  ],
  "detected_language": "en"
}
```

## Notes

- If `diarization` is enabled, a valid `huggingface_access_token` is required.
- The `language_detection_min_prob` and `language_detection_max_tries` parameters are applicable only when `language` is not specified.
- Adjust `batch_size` according to your GPU memory for optimal performance.

## Acknowledgments

- Portions of this project utilize code from [WhisperX](https://github.com/m-bain/whisperX), licensed under the BSD-2-Clause license.

For any issues or feature requests, please open an issue in this repository.
