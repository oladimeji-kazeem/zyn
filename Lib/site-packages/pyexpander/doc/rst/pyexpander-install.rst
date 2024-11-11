Installing pyexpander
=====================

Parts of pyexpander
-------------------

pyexpander consists of scripts, python modules, documentation and configuration
files. 

pyexpander is available on `pypi <https://pypi.python.org/pypi>`_, as a debian
or rpm package, as a tar.gz and zip file. The following chapters describe how 
to install pyexpander.

Requirements
------------

pyexpander requires python version 3 or newer.

pyexpander is tested on `debian <https://www.debian.org>`_ and 
`Fedora <https://getfedora.org>`_ linux distributions but should run on all
linux distributions. It probably also runs on other flavours of unix, probably
even MacOS, but this is not tested.

It may run on windows, escpecially the `Cygwin <https://www.cygwin.com>`_
environment, but this is also not tested.

Install from pypi with pip
--------------------------

In order to install pyexpander with `pip <https://en.wikipedia.org/wiki/Pip_(package_manager)>`_, 
you use the command [1]_::

  pip3 install pyexpander

.. [1] Your version of pip may have a different name, e.g. "pip-3", "pip-3.2" or just "pip"

You find documentation for the usage of pip at `Installing Python Modules
<https://docs.python.org/3/installing/index.html#installing-index>`_.

Install from a debian package
-----------------------------

There are packages for some of the recent debian versions. In order to see
what debian version you use enter::

  cat /etc/debian_version

Download the package here:

* `pyexpander downloads at Sourceforge <https://sourceforge.net/projects/pyexpander/files/?source=navbar>`_

and install with::

  dpkg -i <PACKAGENAME>

The packages may with other debian versions or debian package based
distributions like ubuntu, but this is not tested. 

Install from a rpm package
--------------------------

There are packages for some of the recent fedora versions. 
In order to see what fedora version you use enter::

  cat /etc/fedora-release

Download the package here:

* `pyexpander downloads at Sourceforge <https://sourceforge.net/projects/pyexpander/files/?source=navbar>`_

and install with::

  rpm -ivh  <PACKAGENAME>

The packages may work with other fedora versions or rpm package based
distributions like, redhat, scientific linux or opensuse, but this was not
tested. 

Install from source with setup.py
---------------------------------

You should do this only if it is impossible to use one of the methods described
above. 

Download the tar.gz or zip file here:

* `pyexpander downloads at Sourceforge <https://sourceforge.net/projects/pyexpander/files/?source=navbar>`_

unpack the tar.gz file with::

  tar -xzf <PACKAGENAME>

or unpack the zip file with::

  unzip <PACKAGENAME>

The pyexpander distribution contains the install script "setup.py". If you install
pyexpander from source you always invoke this script with some command line options. 

The following chapters are just *examples* how you could install pyexpander. For a
complete list of all possibilities see 
`Installing Python Modules
<https://docs.python.org/3/installing/index.html#installing-index>`_.

In order to install the program use::

  python3 setup.py [options]

Install as root to default directories
++++++++++++++++++++++++++++++++++++++

This method will install pyexpander on your systems default python library and
binary directories.

Advantages:

- You don't have to modify environment variables in order to use pyexpander.
- All users on your machine can easily use pyexpander.

Disadvantages:

- You must have root or administrator permissions to install pyexpander.
- Files of pyexpander are mixed with other files from your system in the same
  directories making it harder to uninstall pyexpander.

For installing pyexpander this way, as user "root" enter::

  python setup.py install

Install to a separate directory
+++++++++++++++++++++++++++++++

In this case all files of pyexpander will be installed to a separate directory.

Advantages:

- All pyexpander files are below a directory you specify, making it easy to uninstall
  pyexpander.
- If you have write access that the directory, you don't need root or
  administrator permissions.

Disadvantages:

- Each user on your machine who wants to use pyexpander must have the correct
  settings of the environment variables PATH and PYTHONPATH.

For installing pyexpander this way, enter::

  python setup.py install --prefix <DIR>

where <DIR> is your install directory.

In order to use pyexpander, you have to change the environment variables PATH and
PYTHONPATH. Here is an example how you could do this::

  export PATH=<DIR>/bin:$PATH
  export PYTHONPATH=<DIR>/lib/python<X.Y>/site-packages:$PYTHONPATH

where <DIR> is your install directory and <X.Y> is your python version number.
You get your python version with this command::

  python -c 'from sys import *;stdout.write("%s.%s\n"%version_info[:2])'

You may want to add the environment settings ("export...") to your shell setup,
e.g. $HOME/.bashrc or, if your are the system administrator, to the global
shell setup.

Install in your home
++++++++++++++++++++

In this case all files of pyexpander are installed in a directory in your home called
"pyexpander".

Advantages:

- All pyexpander files are below $HOME/pyexpander, making it easy to uninstall pyexpander.
- You don't need root or administrator permissions.

Disadvantages:

- Only you can use this installation.
- You need the correct settings of environment variables PATH and
  PYTHONPATH.

For installing pyexpander this way, enter::

  python setup.py install --home $HOME/pyexpander

You must set your environment like this::

  export PATH=$HOME/pyexpander/bin:$PATH
  export PYTHONPATH=$HOME/pyexpander/lib/python:$PYTHONPATH

You may want to add these lines to your shell setup, e.g. $HOME/.bashrc.

