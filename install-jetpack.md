# Download and Install JetPack (for TX2 only)

This document is intended to help you get familiar with installing JetPack, using the tools, and running sample code.

## System Requirements

Host Platform:

* Ubuntu Linux x64 v14.04 or v16.04

  Note that a valid Internet connection and at least 10GB of disk space is needed for the complete installation of JetPack.

Target Platform:

* Jetson TX2 Developer Kit
* Additional target requirements: 
  * USB Micro-B cable connecting Jetson to your Linux host for flashing.
  * _\(Not included in the developer kit\)_ To connect USB peripherals such as keyboard, mouse, and \[optional\] USB/Ethernet adapter \(for network connection\), a USB hub could be connected to the USB port on the Jetson system.
  * An HDMI cable plugged into the HDMI port on Jetson Developer Kit, which is connected to an external HDMI display.
  * An Ethernet cable plugged into the on-board Ethernet port, which is connected to either a secondary network card on your Linux host or the same network router providing internet access for the Linux host.

## Download the Latest JetPack Version

The latest version of JetPack is available in the NVIDIA Embedded Developer Zone at:

> [https://developer.nvidia.com/jetson-development-pack](https://developer.nvidia.com/jetson-development-pack)

All available JetPack downloads can be found at:

> [https://developer.nvidia.com/jetpack-archive](https://developer.nvidia.com/jetpack-archive)

**Downloading JetPack L4T**

* On the host machine running Ubuntu, create a new directory to store installation packages.
* Download `JetPack-${VERSION}.run` into the new directory on the host Ubuntu machine.

{% hint style="info" %}
Avoid running or installing JetPack in a path that contains a "." Paths that contain a "." are known to cause installation issues.
{% endhint %}

## Installing JetPack L4T

JetPack L4T runs on the host Ubuntu x86\_64 machine and sets up your development environment and Jetson Development Kit target via remote access. Please refer to the [System Requirements](https://docs.nvidia.com/jetpack-l4t/content/developertools/mobile/jetpack/l4t/3.2rc/jetpack_l4t_install.htm?tocpath=_____3#System_Requirements) section for supported hardware configurations.

The following instructions assume you have downloaded the latest JetPack version, `JetPack-${VERSION}.run`, where `${VERSION}` refers to the version string for the installer you have.

1.  Add exec permission for the `JetPack-${VERSION}.run`

```text
chmod +x JetPack-${VERSION}.run
```

2. Run `JetPack-${VERSION}.run` in terminal on your host Ubuntu machine.

![img](https://docs.nvidia.com/jetpack-l4t/content/developertools/mobile/jetpack/images/jetpack_l4t_install.006_600x441.png)

3. Next, the JetPack installer will indicate the installation directory.

![img](https://docs.nvidia.com/jetpack-l4t/content/developertools/mobile/jetpack/images/jetpack_l4t_directory.006_600x441.png)

4. Select the development environment to setup.

![img](https://docs.nvidia.com/jetpack-l4t/content/developertools/mobile/jetpack/images/jetpack_l4t_dev_environment_tx2_only.001_600x441.png)

5. The JetPack installer will pop up a window to ask for permission to use during the installation process; you will need to enter your sudo password here.

```text
  ![img](https://docs.nvidia.com/jetpack-l4t/content/developertools/mobile/jetpack/images/jetpack_l4t_sudo_pw.001_450x210.png)
```

6. The Component Manager opens, which allows you to customize which components to install. Select the Jetson Developer Kit you would like to develop for to customize the installation components for each device.

![img](https://docs.nvidia.com/jetpack-l4t/content/developertools/mobile/jetpack/images/jetpack_l4t_component_mgr.009_600x524.png)

{% hint style="info" %}
**NOTE:** To run a standalone Ubuntu install, deselect Jetson target specific entries.
{% endhint %}

7. Accept the license agreement for the selected components.

![img](https://docs.nvidia.com/jetpack-l4t/content/developertools/mobile/jetpack/images/jetpack_l4t_license.004_500x455.png)

8. The Component Manager will proceed with the installation. Once the host installation steps are completed, click the Next button to continue with the installation of target components.

![img](https://docs.nvidia.com/jetpack-l4t/content/developertools/mobile/jetpack/images/jetpack_l4t_begin_target_install.004_600x441.png)

{% hint style="info" %}
**NOTE:** JetPack will now proceed with setting up the Jetson Developer Kit target, if the corresponding components were selected \(i.e., flashing the OS and pushing components to the Jetson Developer Kit target\).
{% endhint %}

9. If you de-selected Flash OS in the Component Manager, you will need to enter the IP address, user name, and password to set up an ssh connection to the target device.

![img](https://docs.nvidia.com/jetpack-l4t/content/developertools/mobile/jetpack/images/jetpack_l4t_device_info.007_600x441.png)

After you enter the required information and click **Next**, JetPack will begin installing components on the target device.

10. If you selected Flash OS in the Component Manager, you will need to select the network layout for your specific environment.

![img](https://docs.nvidia.com/jetpack-l4t/content/developertools/mobile/jetpack/images/jetpack_l4t_network_layout.008_600x441.png)

11. If you selected the Device access Internet via router/switch layout, you will be asked to select which interface to use for Internet access.

![img](https://docs.nvidia.com/jetpack-l4t/content/developertools/mobile/jetpack/images/jetpack_l4t_network_interface.007_600x441.png)

12. If you selected the Device get IP assigned by DHCP server on host and access Internet via host machine layout, you must select which interface is to be used for Internet access, and which is to be used for the target interface.

![img](https://docs.nvidia.com/jetpack-l4t/content/developertools/mobile/jetpack/images/jetpack_l4t_network_selection.005_600x441.png)

13. A pop-up window will instruct you to put your device into Force USB Recovery Mode, so you can flash the OS.

![img](https://docs.nvidia.com/jetpack-l4t/content/developertools/mobile/jetpack/images/jetpack_l4t_force_recovery_mode.001_600x364.png)



14. Next, you will be prompted to install components on the specific target machine, and to compile samples.

![img](https://docs.nvidia.com/jetpack-l4t/content/developertools/mobile/jetpack/images/jetpack_l4t_install_complete.006_600x441.png)

15. After the post installation tasks have been completed, the installation will be complete.

![img](https://docs.nvidia.com/jetpack-l4t/content/developertools/mobile/jetpack/images/jetpack_l4t_post_install.007_600x441.png)

## Compiling Samples

JetPack automatically compiles all samples, if **Compile Samples** was checked during the component selection portion of the installation. CUDA samples can be found in the following directory:

```text
 <JetPack_Install_Dir>/NVIDIA_CUDA-<version>_Samples
```

You can recompile the samples by running:

```text
SMS=53 EXTRA_LDFLAGS=--unresolved-symbols=ignore-in-shared-libs TARGET_ARCH=aarch64 make
```

## Run Sample Code

The CUDA samples directory will be copied to the home directory on your device by JetPack. The built binaries are in the following directory:

```text
/home/ubuntu/NVIDIA_CUDA-<version>_Samples/bin/aarch64/linux/release/
```

Run them by calling them in terminal, or double-clicking on them in the file browser. For example, when you run the oceanFFT sample, the following screen will be displayed.

![img](https://docs.nvidia.com/jetpack-l4t/content/developertools/mobile/jetpack/images/jetpack_cuda_fft_ocean_simulation.001_500x527.png)

