<img src="/public/logo.svg" width="100" height="100">

# LL90: Leo's LeetCode 90

Hello and welcome to LeetCode 90, a personal project of mine that aims to streamline your (or my) journey through LeetCode's coding challenges.

I created this list of 90 questions to maximize learning efficiency by:
* spreading questions on the same topic unless they are closely related,
* increasing the difficulty gradually but not strictly,
* and ensuring a good coverage of topics even if the whole list is not completed.

## Target Audience

This list is primarily designed for those getting ready for interviews in roles adjacent to Software Engineering, but not exactly Software Engineering. This includes positions like Research Engineer, Research Scientist, Machine Learning Engineer, Computer Vision Engineer, and so on.

While this list overlaps with popular lists like Grind75, Neetcode Roadmap, and LeetCode Top Interview 150, it's uniquely tailored. In addition to the classics, this list focuses on questions that are more practical and relevant for the mentioned roles. These aren't overly difficult questions, but rather ones that are genuinely useful and often implemented by professionals in these fields time to time.

## Features

* **Strategically Organized Questions:** The list is organized to maximize learning efficiency.
* **Progress Tracking:** Automatically saves your progress, so you can pick up right where you left off.
* **Portable Progress:** Your progress can be downloaded and uploaded, so switching devices is seamless.
* **Solutions Included:** I tried to keep them as clean and easy to follow as possible. Some of them also have some personal fun-facts.

## Project Structure

- `generator`: Contains the scripts for turning markdown into web pages.
- `generator/paperplane.py`: This is a super-simple static website generator that I built a long time ago (2015). To this date, I still use it in my personal projects including [my homepage](http://www.isikdogan.com).
- `generator/content.md`: This is where all the questions and their solutions are.
- `public`: Hosts the frontend - HTML, CSS, and JS.

To update the `index.html`, simply `cd generator`, update `content.md`, and run `paperplane.py`.

## Tech Stack

* Python: Used for backend scripting and page generation.
* Jinja2: Templating engine for rendering HTML pages.
* Markdown: For writing and formatting content.
* Bootstrap: For responsive and modern UI.
* JavaScript: For interactive front-end components.
* Prism.js: For syntax highlighting.
* Font Awesome: For icons.

## Contributions 

Contributions are welcome! If you have suggestions or want to improve the project, feel free to open an issue or submit a pull request

## Fun Fact

One of the reasons I created this project was to find intrinsic motivation for grinding LeetCode. This project helped me change my mindset from preparing for interviews to building a product that people might love. Preparing for interviews sucks, but building products is fun!
