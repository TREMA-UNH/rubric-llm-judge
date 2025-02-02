{
  description = "rubric-llm-judge";

  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-24.05";
  inputs.nixpkgs.follows = "dspy-nix/nixpkgs";
  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.dspy-nix.url = "git+https://git.smart-cactus.org/ben/dspy-nix";
  inputs.dspy-nix.follows = "exampp/dspy-nix";
# inputs.exampp.url = "git+ssh://git@git.smart-cactus.org/ben/exampp.git";
  inputs.exampp.url = "git+https://github.com/laura-dietz/rubric-internal";

  outputs = inputs@{ self, nixpkgs, flake-utils, dspy-nix, exampp, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        mkShell = target: (dspy-nix.lib.${system}.mkShell {
          inherit target;
          pythonOverrides = [ exampp.lib.${system}.pythonOverrides ];
          pythonPackages = ps: [ ps.exampp ps.scikit-learn ps.mypy ps.pylatex];
        });

        pythonOverrides = pkgs.lib.composeOverlays
          exampp.lib.${system}.pythonOverrides
          (self: super: {
            rubric_llm_judge = self.buildPythonPackage {
              name = "rubric_llm_judge";
              src = ./.;
              format = "pyproject";
              propagatedBuildInputs = with self; [
                setuptools
                pydantic
                exampp
              ];
            };
          });

      in {
        lib.pythonOverrides = pythonOverrides;
        packages.exampp = (pkgs.python3.override {
          packageOverrides = exampp.lib.${system}.pythonOverrides;
        }).pkgs.exampp;

        packages.rubric_llm_judge = (pkgs.python3.override {
          packageOverrides = pythonOverrides;
        }).pkgs.rubric_llm_judge;

        devShells.default = self.outputs.devShells.${system}.cuda;
        devShells.cpu = mkShell "cpu";
        devShells.rocm = mkShell "rocm";
        devShells.cuda = mkShell "cuda";
      }
    );
  nixConfig = {
    substituters = [ "https://cache.nixos.org" "https://dspy-nix.cachix.org" ];
    trusted-public-keys = [ "cache.nixos.org-1:6NCHdD59X431o0gWypbMrAURkbJ16ZPMQFGspcDShjY=" "dspy-nix.cachix.org-1:VJ553D0iJVoA8ov2+ly+dLnGHarfSQpemzVW6dY6CfE=" ];
  };
  }
