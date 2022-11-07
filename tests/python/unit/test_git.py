import subprocess

import pytest

from ai_art.git import check_lfs, get_branches, shallow_clone


class TestCheckLFS:
    def test_installed(self, mocker):
        """Do nothing when git-lfs is installed"""
        run = mocker.patch("subprocess.run")
        check_lfs()
        run.assert_called_once()

    def test_not_installed(self, mocker):
        """Raise RuntimeError when git-lfs isn't installed"""
        run = mocker.patch("subprocess.run")
        run.side_effect = subprocess.CalledProcessError(1, "failed-command")

        with pytest.raises(RuntimeError):
            check_lfs()


class TestGetBranches:
    _LS_REMOTE_OUTPUT = (
        "4e1243bd22c66e76c2ba9eddc1f91394e57f9f83	refs/heads/branch1\n"
        "4e1243bd22c66e76c2ba9eddc1f91394e57f9f83	refs/heads/extra/level\n"
        "NON-MATCHING LINE\n"
        "4e1243bd22c66e76c2ba9eddc1f91394e57f9f83	refs/not-heads/fake\n"
        "4e1243bd22c66e76c2ba9eddc1f91394e57f9f83	extrarefs/heads/fake\n"
        "9054fbe0b622c638224d50d20824d2ff6782e308   refs/heads/branch2\n"
        "  41c5985fc771b6ecfe8feaa99f8fa9b77ac7d6ce	refs/heads/branch3  \n"
    )

    def test_get_branches(self, mocker):
        run = mocker.patch("subprocess.run")
        run.return_value.stdout = self._LS_REMOTE_OUTPUT

        branches = get_branches("REPO", username="USER", password="PASS")

        assert tuple(branches) == ("branch1", "branch2", "branch3")
        run.assert_called_once()


class TestShallowClone:
    def test_clone(self, mocker):
        run = mocker.patch("subprocess.run")
        shallow_clone(
            "REPO", "DIR", branch="BRANCH", username="USER", password="PASS"
        )
        run.assert_called_once()
