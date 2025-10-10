# Reveal.js template

This repository provides a ready-to-use template for writing slides in Markdown using 
[Reveal.js](https://revealjs.com/) and hosting them via GitHub Pages or by running
a local server.

Reveal.js is an open-source framework for creating HTML-based presentations using 
Markdown or HTML.
Instead of writing slides in a visual editor (like Keynote or PowerPoint), you write 
them in plain text — and Reveal.js turns them into a modern, interactive slideshow in 
the browser.


## 🚀 Getting Started

### 1. Install Node.js and npm

You need Node.js (which includes npm) installed on your system.

#### Windows / macOS / Linux:
- Visit: https://nodejs.org/
- Download the **LTS** version and install it

To verify:
```bash
node -v
npm -v
```

### 2. Get the Template

**Do not clone this repository.**  
Instead:

1. Visit: https://github.com/arampatzis/reveal-template
2. Click **Code → Download ZIP**
3. Extract it on your computer
4. Rename the folder as you wish

### 3. Install Required Packages

Inside the extracted folder, open a terminal and run:

```bash
npm install
```

This installs local development tools like the server (`gulp`).


### 4. Run the Presentation Locally

```bash
npm start
```

Open your browser at:
```
http://localhost:8000
```

You should see your slides rendered.


## 📝 Editing Your Slides

Slides are written in `slides.md` using Markdown.

Use `---` to separate horizontal slides, and `--` for vertical slides:

```markdown
# Welcome

---

## Slide 1

Some text

--

- Vertical slide 1

--

- Vertical slide 2
```


## 🎨 Customizing Theme

Reveal.js themes are in the `dist/theme` folder.

To change the theme, edit `index.html` and modify this line:

```html
<link rel="stylesheet" href="dist/theme/black.css" id="theme">
```

Try other themes like `white.css`, `night.css`, `moon.css`, etc.


## 🚀 Deployment to GitHub Pages

This project uses **GitHub Actions** to automatically deploy the Reveal.js slides —
including `.mp4` videos tracked with **Git LFS** — to GitHub Pages.

### 🛠️ Deployment Workflow

GitHub Actions runs automatically on every push to the `main` branch.
It performs the following steps:

1. Checks out the repo with Git LFS support
2. Pulls all Git LFS content (e.g., `.mp4` files)
3. Copies the site contents (including videos) to a `public/` folder
4. Publishes the `public/` folder to the `gh-pages` branch via [peaceiris/actions-gh-pages](https://github.com/peaceiris/actions-gh-pages)


### ✅ Enable GitHub Pages

After you push your slides and the workflow was successful, a branch called `gh-pages` 
will be created. Follow these steps to enable GitHub Pages:

- GitHub Pages must be enabled:
  - Go to **Settings → Pages**
  - Set **Source** to: `gh-pages` branch, `/ (root)`
- Your `.mp4` files should be downloaded via Git LFS during the workflow



This allows you and your students to share presentations with a link.


## 🔒 Public vs Private Repos and Ways to Share Slides

### ✅ Option 1: GitHub Pages (Public Repo)
- If your repository is **public**, GitHub Pages will host your presentation at:
  ```
  https://<your-username>.github.io/<repo-name>/
  ```
- This is the easiest and recommended way to publish Reveal.js slides.

### 🚫 Option 2: GitHub Pages (Private Repo)
- GitHub Pages **does not** work for private repos unless you are on a **GitHub Enterprise** plan.
- You will see a 404 or the site won’t build.

### 🛠 Option 3: Local Server (for Private Sharing)
- You can run your slides locally with:
  ```bash
  npm start
  ```
- Then access them at:
  ```
  http://localhost:8000/
  ```
- This works for development or sharing over a local network.

### 📤 Option 4: Export to PDF
- Reveal.js can export slides to PDF:
  ```bash
  npm install -g decktape
  decktape reveal http://localhost:8000 slides.pdf
  ```
- Useful for email, offline viewing, or submissions.

These options allow flexibility depending on your privacy needs.


## 📦 Managing Large Files with Git LFS

If you need to include large binary files in your presentation (e.g. `.mp4` videos, `.pdf` documents), it is recommended to use **Git Large File Storage (LFS)** instead of committing them directly to the repository.

### ✅ Install Git LFS

- **macOS:**  
  `brew install git-lfs`

- **Ubuntu/Debian:**  
  `sudo apt install git-lfs`

- **Windows:**  
  Download and install from [git-lfs.com](https://git-lfs.com)

### ✅ Set Up Git LFS in Your Repo

1. Run this once (per machine):
   ```bash
   git lfs install
   ```

2. Track the file types you want to store with LFS:
   ```bash
   git lfs track "*.mp4"
   git lfs track "*.pdf"
   ```

3. Add and commit:
   ```bash
   git add .gitattributes
   git add media/my-video.mp4
   git commit -m "Add video using Git LFS"
   git push
   ```

### ℹ️ Notes

- GitHub’s free LFS plan includes **1 GB of storage** and **1 GB/month bandwidth**.
- You can view tracked LFS files using:
  ```bash
  git lfs ls-files
  ```

To avoid LFS entirely, you can also upload media externally (e.g. Google Drive or a web server) and link to them in your slides.
