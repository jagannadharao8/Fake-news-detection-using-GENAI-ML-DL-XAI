# Contributing to the Fake News Detection System

First off, thank you for considering contributing to the Fake News Detection System! It's people like you that make the open-source community such an amazing place to learn, inspire, and create.

## Where do I go from here?

If you've noticed a bug or have a feature request, make sure to check if there's already an [issue](https://github.com/jagannadharao8/Fake-news-detection-using-GENAI-ML-DL-XAI/issues) open for it. If not, feel free to open one!

## Fork & create a branch

If this is something you think you can fix, then fork the repository and create a branch with a descriptive name.

A good branch name would be (where issue #325 is the ticket you're working on):

```sh
git checkout -b 325-add-new-model
```

## Get the test suite running

Make sure your local environment is correctly set up. You can refer to the `README.md` for installation and configuration instructions. Ensure that your changes do not break any existing functionality.

## Implement your fix or feature

At this point, you're ready to make your changes. Feel free to ask for help; everyone is a beginner at first.

## Make a Pull Request

At this point, you should switch back to your master branch and make sure it's up to date with the main repository's master branch:

```sh
git remote add upstream https://github.com/jagannadharao8/Fake-news-detection-using-GENAI-ML-DL-XAI.git
git checkout master
git pull upstream master
```

Then update your feature branch from your local copy of master, and push it!

```sh
git checkout 325-add-new-model
git rebase master
git push --set-upstream origin 325-add-new-model
```

Finally, go to GitHub and make a Pull Request.

## Keeping your Pull Request updated

If a maintainer asks you to "rebase" your PR, they're saying that a lot of code has changed, and that you need to update your branch so it's easier to merge.

Thank you for contributing!
