#!/usr/bin/env bash
# Packages the twitter_facebook_tiktok dataset directory into a zip archive for
# sharing or upload. Excludes macOS .DS_Store files. After zipping, lists the
# archive contents so you can verify the result. Run from the repo root.

pushd ./scratch/datasets/
zip -r twitter_facebook_tiktok.zip twitter_facebook_tiktok -x '**/.DS_Store'
echo ""
unzip -l twitter_facebook_tiktok.zip
popd
