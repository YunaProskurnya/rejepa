1. Follow KISS everywhere. Less is more.
2. Don't touch existed comments.
3. Don't write comments in code, but devide with space logical block of code
4. If you don't write tests, use as little functions and classes as needed

# Remote GPU Server for `/Users/superustam` User

- Local code: `/Users/superustam/XProject/rejepa`
- Remote code: `/home/container/rejepa`
- Remote data and outputs: `/home/container/shared/rejepa`

Use `ssh skoltech_container` to execute remote GPU experiments.
Use `rsync` to synchronize code.

Always:
- Run training in tmux.
- Save experiment logs.
- Check GPU availability before running training.
- Never terminate other users' processes.