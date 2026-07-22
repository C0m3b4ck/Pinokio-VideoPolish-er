module.exports = {
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "python cli.py --help",
        ],
        input: true,
        chain: true,
      }
    },
  ]
}