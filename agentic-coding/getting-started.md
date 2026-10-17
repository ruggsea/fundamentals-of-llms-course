# Getting started with coding agents

Use **Pi** and choose a model subscription below. The *harness* is the app that reads files, edits code and runs commands, the *provider* supplies the AI model. Pi and OpenCode are free harnesses. Copilot and OpenCode Go provide the model access.


## Choose your subscription

| Option | Cost | How you connect to Pi | Model choice |
| --- | --- | --- | --- |
| GitHub Copilot Student | Free for verified, eligible students | Lorenz’s plugin and GitHub browser sign-in | Auto chooses an available model |
| OpenCode Go | US$10/month at the advertised price | Your own OpenCode API key | Choose from the Go models |

Start with Copilot if you already have the student benefit. Go is an alternative if you want to choose and have better models. Both have usage limits. See the current [Copilot plans](https://docs.github.com/en/copilot/get-started/plans) and [Go plans and limits](https://opencode.ai/v2/docs/console/go/) before subscribing.

## Install Pi once

Install [Node.js](https://nodejs.org/en/download) **22.19 or newer** and [Git](https://git-scm.com/downloads). Open a terminal and run:

```sh
node --version
git --version
npm install -g --ignore-scripts @earendil-works/pi-coding-agent@1.2.0
pi --version
```

If `pi` is not found, reopen the terminal. For alternative installers, see [Pi’s installation instructions](https://github.com/earendil-works/pi#getting-started).

The commands in terminal code blocks run in your terminal. Commands beginning with `/`, such as `/login`, are typed **inside Pi**.

## Option A GitHub Copilot Student

1. Open your [GitHub Education benefits](https://github.com/settings/education/benefits). Apply for student verification if needed, then activate **Copilot Student**. Verification and activation are separate steps, and activation can take several days. Follow [GitHub’s student instructions](https://docs.github.com/en/copilot/how-tos/copilot-on-github/set-up-copilot/enable-copilot/set-up-for-students), you do not need to buy a paid plan to claim the student benefit.
2. Install [Lorenz’s Pi Copilot Student plugin](https://github.com/lorenzprattes/pi-copilot-student):

   ```sh
   pi install git:github.com/lorenzprattes/pi-copilot-student@v0.1.1
   pi
   ```

3. Inside Pi, enter `/login github-copilot`. Choose browser/device sign-in, leave the Enterprise domain blank for a normal github.com account, and authorize the GitHub account with your student benefit.
4. Enter `/model` and choose **Copilot Auto** (`github-copilot-auto/auto`). Student access requires Auto, selecting a named model under the ordinary Copilot provider will fail.
5. Enter `/copilot-student-doctor` to check the connection. If you installed the plugin while Pi was open, restart Pi first.

The plugin is a community integration, not an official GitHub client. It worked in our workshop setup, but changes to GitHub’s private endpoints may require a plugin update. No API key needs to be copied for this option.

## Option B OpenCode Go

An **API key** is a secret credential that lets Pi use your subscription.

1. Sign in to the [OpenCode Console](https://opencode.ai/console/). Use your **Personal workspace**, subscribe to **Go**, and check the price at checkout.
2. Create a key in that workspace’s **Keys** area. In the newer console, the documented sequence is **Add Service Account** → name it `Pi` → **Add API Key** → choose **Inference only** → **Create key**. Copy the key when shown, it is displayed only once. A service account is a credential for an application. See [OpenCode’s key instructions](https://opencode.ai/v2/docs/console/api/#service-accounts).
3. Start `pi`. Inside Pi, enter `/login opencode-go` and paste the key into the **API-key prompt**, not into an ordinary chat message. Pi saves it locally. No plugin or hand-edited configuration is needed. See [Pi authentication](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/providers.md).
4. Enter `/model` and choose a model under **OpenCode Go** (`opencode-go`), for example **GLM-5.3-Flash** if available. Be sure you selected Go: the separate OpenCode/Zen provider uses different billing.

Check usage in the Console. Leave **Use balance / Extra usage** disabled if you want requests to stop at the subscription limit rather than consume additional paid credits. The [Go documentation](https://opencode.ai/v2/docs/console/go/#usage-beyond-limits) explains this option and lists available models.

## Start your first coding session

Open a terminal **in your project folder**, or navigate there first:

```sh
cd path/to/your/project
pi
```

Use `/model` to select your configured provider. Check the selected model in the footer before sending a task. 

Generally, document well what you desire from the piece of software you want to produce. Clear specification requires a lot of thinking, and this in turn is what is required for good results from agentic coding.

## If you already have an existing project:

Start with:

> Read the README and explain what this project does and how to run its tests. Do not change files yet.

Then give one small, checkable task:

> Add a test for empty input. Explain the expected behavior first, make the smallest change, and run the tests. Do not weaken existing tests.

Review the changes with `git diff` and run the tests yourself. Ask the agent to explain code you do not understand.

Useful Pi controls: **Esc** stops the current generation, `/new` starts a fresh conversation, `/resume` reopens a saved session. In `/model`, **Ctrl+S** saves the selected model as your default. See [Pi usage](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/usage.md).



## Keep these habits

- **Work in Git repos, ideally with worktrees.** Pi can edit files and execute shell commands without asking permission by default. Use a container or virtual machine for unfamiliar projects.
- **Keep keys and private data out of chat and Git.** Never share your API key or Pi’s `~/.pi/agent/auth.json`. Hosted models receive the prompts and file content the agent sends them.
- **If setup fails:** for Copilot, run `/copilot-student-doctor` and confirm Student activation. For Go, check the workspace, subscription and key, then repeat `/login opencode-go`. For quota errors, check the provider’s usage page. 
