# Contributing

Thanks for considering a contribution to the Social Media Sentiment Analyzer.

## Getting started
1. Fork and clone the repository.
2. Create a Python virtual environment and install `requirements.txt`.
3. Follow the README to train the model, run predictions and launch the dashboard.

## Contribution guidelines
- Keep preprocessing behaviour for emojis, hashtags, negation and Hinglish in mind.
- Avoid data leakage: model selection must not use the held-out test set.
- Add tests for changes to preprocessing, training, prediction or dashboard behaviour.
- Run `pytest` before submitting a pull request.
- Do not commit secrets, personal datasets or generated local artifacts unless they are intended project assets.

## Pull requests
Describe the issue, the implementation, test results and any change in model evaluation. Include relevant screenshots for dashboard changes.
