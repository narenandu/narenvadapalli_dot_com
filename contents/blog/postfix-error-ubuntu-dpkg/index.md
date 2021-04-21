---
title: Fixing the postfix error dpkg
date: 2021-04-21
template: blog
image: ./image.png
banner: ./banner.jpg
description: Steps to fix the postfix error happening during apt upgrade ubuntu.
---

#### Problem Statement

During the upgrade process from **Ubuntu 20.10** to **Ubuntu 21.04**, while running

```
sudo apt upgrade -y
```

Was hitting the error consistently about `postfix`

```
/home/naren   $ sudo apt dist-upgrade
[sudo] password for naren:
Reading package lists... Done
Building dependency tree... Done
Reading state information... Done
Calculating upgrade... Done
0 upgraded, 0 newly installed, 0 to remove and 0 not upgraded.
1 not fully installed or removed.
After this operation, 0 B of additional disk space will be used.
Do you want to continue? [Y/n] y
Setting up postfix (3.5.6-1) ...
setting myhostname=naren-PC.. in /etc/postfix

Postfix (main.cf) is now set up with the changes above.  If you need to make
changes, edit /etc/postfix/main.cf (and others) as needed.  To view Postfix
configuration values, see postconf(1).

After modifying main.cf, be sure to run 'systemctl reload postfix'.

Running newaliases
newaliases: warning: valid_hostname: misplaced delimiter: naren-PC..
newaliases: fatal: file /etc/postfix/main.cf: parameter myhostname: bad parameter value: my-PC..
dpkg: error processing package postfix (--configure):
 installed postfix package post-installation script subprocess returned error exit status 75
Processing triggers for libc-bin (2.33-0ubuntu5) ...
Errors were encountered while processing:
 postfix
E: Sub-process /usr/bin/dpkg returned an error code (1)

```

#### Solution

-   Observe the `value: my-PC..` in the error
-   Open up the `/etc/postfix/main.cf` in an editor with `sudo` mode

```
sudo vim /etc/postfix/main.cf
```

-   Find the line that has the following

```
myhostname = my-PC..
```

-   Replace with the following

```
myhostname = my-PC.name
```

-   Reload the `postfix` service

```
etc/init.d/postfix reload
```

Should show the following

```
$ /etc/init.d/postfix reload
Reloading postfix configuration (via systemctl): postfix.service.
```

Now when you run `sudo apt upgrade -y`, we should have no errors !

#### Fixed Run

```
$ sudo apt upgrade
...
Hit:2 http://security.ubuntu.com/ubuntu hirsute-security InRelease
Hit:3 http://ca.archive.ubuntu.com/ubuntu hirsute InRelease
Hit:4 http://ca.archive.ubuntu.com/ubuntu hirsute-updates InRelease
Hit:5 http://ca.archive.ubuntu.com/ubuntu hirsute-backports InRelease
Reading package lists... Done
Building dependency tree... Done
Reading state information... Done
All packages are up to date.
Reading package lists... Done
Building dependency tree... Done
Reading state information... Done
Calculating upgrade... Done
0 upgraded, 0 newly installed, 0 to remove and 0 not upgraded.
1 not fully installed or removed.
After this operation, 0 B of additional disk space will be used.
Do you want to continue? [Y/n] y
Setting up postfix (3.5.6-1) ...

Postfix (main.cf) configuration was not changed.  If you need to make changes,
edit /etc/postfix/main.cf (and others) as needed.  To view Postfix
configuration values, see postconf(1).

After modifying main.cf, be sure to run 'systemctl reload postfix'.

Running newaliases
Created symlink /etc/systemd/system/multi-user.target.wants/postfix.service → /lib/systemd/system/postfix.service.
Processing triggers for libc-bin (2.33-0ubuntu5) ...
```
