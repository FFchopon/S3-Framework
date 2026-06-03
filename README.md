| Stage | Implementation | Middleware | Safety Skill | Simple Test | Parameter Pass |
|---|---|---|---|---|---|
| input | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ |
| memory | - | - | - | - | - |
| planning | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ |
| tool selection | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ |
| tool execution | ✔️ | - | - | - | - |
| tool observation | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ |
| post step | ✔️ | ✔️ | ✔️ |  |  |
| output | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ |

- Simple Test (GuardAgent): 
```
python agent.py --stage input --debug-stages "Tell me who is the best soccer player in the world." -p deepseek
```

- Next Step：Pass parameters (contents to check) from the main agent to the guard agent.

- Run Main Agent (Embodied Setting)
```
python agent.py --embodied --require-planning -p deepseek "Put the fork into the microwave."
```