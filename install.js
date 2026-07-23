module.exports = {
  requires: {
    bundle: "ai"
  },
  run: [
    {
      method: "shell.run",
      params: {
        message: [
          "sudo apt install -y libcublas12",
        ],
      }
    },
    {
      method: "shell.run",
      params: {
        message: [
          "git clone https://github.com/C0m3b4ck/VideoPolish-er.git app",
        ]
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install -r requirements.txt",
        ]
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "python -c \"import faster_whisper; print('faster-whisper OK')\"",
          "python -c \"import pydub; print('pydub OK')\"",
          "python -c \"import soundfile; print('soundfile OK')\"",
        ]
      }
    },
  ]
}