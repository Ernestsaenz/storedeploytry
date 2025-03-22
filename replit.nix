{ pkgs }: {
  deps = [
    pkgs.bash
    pkgs.libxcrypt
    pkgs.python311Full
  ];
  env = {
    PYTHON_LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [ pkgs.python311Full ];
    PYTHONBIN = "${pkgs.python311Full}/bin/python3.11";
    LANG = "en_US.UTF-8";
  };
}
