---
title: Storing Github access token in git credential store
date: 2023-04-04
template: blog
image: ./image.jpg
banner: ./banner.jpg
description: Using git credentials store the github access token to avoid the re-prompting of username and pwd
---


## Motivation
`git` operations were prompting the github access token despite having `GITHUB_ACCESS_TOKEN` environment variable in WSL2 on Windows

## Prerequisites
Go through the process of generating the github access token as mentioned in [Github login using access token via cmdline blog](https://www.narenvadapalli.com/blog/github-login-using-access-token-via-cmdline/)


## Solution
Run the following sequence of steps to seek the help of git's credential store on the machine to store the access token

```
git config --global credential.helper store
```

Run the following command, which will wait for the input from user. So it is expected that you will see a waiting prompt and not a completed command.
```
git credential-store --file ~/.git-credentials store
```

> Essentially you are storing the access token in `~/.git-credentials` file on disk

Now update the actual github access token against the `username` key in the following block and paste it at the waiting command prompt
> leave the `password=` as blank, no need to specify a value
```
protocol=https
host=github.com
username=<access-token>
password=
```

Hit `Enter` twice. This will save the credential and exit the command prompt

----

If you inspect the credential file, you should see something similar as below
```
$ cat ~/.git-credentials
https://ghp_XXXXXXXXXXXXXXXXXXXXXXX:@github.com
```
----

That's it ! From now on, any `git` commands shouldn't prompt you for username (github user) and password (access token)
