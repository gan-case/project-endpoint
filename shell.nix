with import <nixpkgs> { };

let pythonPackages = python310Packages;
in pkgs.mkShell rec {
	name = "mar-fastapi-env";
	venvDir = "./env";
	buildInputs = [
		stdenv.cc.cc.lib
        stdenv.cc
		pythonPackages.python
		pythonPackages.venvShellHook
	];

	LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath buildInputs;

	postVenvCreation = ''
		pip install -r requirements.txt
	'';
}
