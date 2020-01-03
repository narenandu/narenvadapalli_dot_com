---
path: "/blog/safely-deleting-linux-from-dualboot.md"
date: "2020-01-02"
title: "Safely deleting linux from dual boot"
---

- Restart the machine if not in Windows OS
- Select Windows (`/dev/sda`) from the menu
- No need to login
- Just hold SHIFT and press the power icon at the bottom right to see `Restart` in the pop up menu
- Keep on holding SHIFT (might have to leave it for a couple of seconds in the middle) until `Restarting` slowly transitions to `Please Wait`
- Hit `TroubleShoot` option from the list
- Hit `Advanced Options` from the list
- Hit `Command Prompt` from the list
- System reboots to the grub boot loader menu
- Select Windows (`/dev/sda`) from the menu
- Select your user account
- Login with your password
- Type the following sequence of commands
```
bootrec /fixmbr
bootsect/nt60 sys
bootrec /fixboot
bootrec /scanos
bootrec /rebuildbcd
exit
```
- Hit `Turn off your PC`
- Power up the host again
- Select Windows (`/dev/sda`) from the menu
- Press `Windows Key + X` on keyboard
- Select `Disk Management`
- Delete the partitions related to the Linux OS (leave EFI partition)
- Press `Windows Key + X` on keyboard 
- Select `Power Shell Administrator`
- Type the following commands
```
diskpart
list disk
select disk 0    # this would be the hard disk 
list partition  
select partition 2
assign letter=X
exit
X:
cd EFI
rd fedora        # or whatever linux os 
```
