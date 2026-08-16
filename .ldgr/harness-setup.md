# LDGR harness setup

ldgr installs one skill, `ldgr`, into your harness's global skill directory. It routes an agent to the CLI; the CLI describes itself from there.

If the skill is not installed, run `ldgr install` (interactive, human-operated) and select your harness. If your harness is not listed, copy the skill directory into whatever global skill path it reads, or point the agent at the CLI directly — `ldgr` works from any shell without a skill.

An agent that has not been given the skill should start with `ldgr status` (or `ldgr init` if no `.ldgr/ldgr.db` exists) and then run `ldgr workflow`.

LDGR-owned profiles require the paired agentctl/Core release. Run `agentctl discover --json`; if Core compatibility is false, install or roll back both binaries together before starting a loop.

Read `.ldgr/operator-errors.md` for the operator policy and `.ldgr/agent-errors.md` for the agent checkpoint requirements.
