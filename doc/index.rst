AI Art
%%%%%%

.. TODO: add images
.. TODO: add documentation for the recipes
.. TODO: add documentation for CUDA (system requirements and minimum VRAM)

This plugin allows you to generate images from text using
`Stable Diffusion <stable-diffusion-wiki_>`_

How to set up
=============

Download weights
----------------
Before you can use the plugin, you need pre-trained weights. You can download
weights from Hugging Face using the provided macro.

.. warning::
   The CompVis weights available on Hugging Face are licensed under the
   CreativeML OpenRAIL-M license, which restricts usage. You can view the
   license `here <compvis-license_>`_.

   If you don't agree with the license, you can alternatively use your own
   weights and skip this section

#.  Install `Git LFS <git-lfs_>`_ on the DSS server:

    RHEL-based distros:

    .. code-block:: bash

       yum install git-lfs
       git lfs install --system

    Debian-based distros:

    .. code-block:: bash

       apt install git-lfs
       git lfs install --system

    macOS (using `Homebrew <homebrew_>`_):

    .. code-block:: bash

       brew install git-lfs
       git lfs install

#.  Create a `Hugging Face <hugging-face-sign-up_>`_ account if you don't
    already have one.

#.  Create a read-only access token in your
    `account settings <hugging-face-token-settings_>`_. This will be used by DSS
    to access Hugging Face.

#.  Choose the model that you want to download from
    `CompVis' organization page <compvis_>`_, and agree to the license in order
    to access the repository.

    .. note::
       If you're not sure which model to choose, pick the one with the highest
       version, e.g. *stable-diffusion-v1-4*.

#.  Create a managed folder in DSS, and download your chosen model to it using
    the *Download Stable Diffusion weights* macro.

    .. warning::
       The managed folder must be stored on the local filesystem. Folders stored
       on remote connections (Amazon S3, Google Cloud Storage, etc) aren't
       supported.

.. _hugging-face-sign-up: https://huggingface.co/join
.. _hugging-face-token-settings: https://huggingface.co/settings/tokens
.. _compvis: https://huggingface.co/CompVis
.. _compvis-license: https://huggingface.co/spaces/CompVis/stable-diffusion-license
.. _git-lfs: https://git-lfs.github.com/
.. _stable-diffusion-wiki: https://en.wikipedia.org/wiki/Stable_Diffusion
.. _homebrew: https://brew.sh/
