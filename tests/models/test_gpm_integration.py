import pytest

from models.gpm import Gpm


def test_gpm_name_and_compatibility():
    assert Gpm.NAME == "gpm"
    assert "class-il" in Gpm.COMPATIBILITY
    assert "task-il" in Gpm.COMPATIBILITY


@pytest.mark.parametrize("arg_name", [
    "gpm_threshold_base",
    "gpm_threshold_increment",
    "gpm_activation_samples",
])
def test_gpm_parser_defaults(arg_name):
    parser = Gpm.get_parser(None)
    args = parser.parse_args([])
    assert hasattr(args, arg_name)

