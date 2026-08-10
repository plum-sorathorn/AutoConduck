from autoconduck.tui.keymap import FOOTER_HINT, KEYMAP, QUIT_KEY


def test_shared_keymap_bindings():
    assert KEYMAP["ctrl+c"][0] == "quit"
    assert not ("q" in KEYMAP and KEYMAP["q"][0] == "quit")
    assert QUIT_KEY == "ctrl+c"
    assert "[ctrl+c]" in FOOTER_HINT()
    assert KEYMAP["down"][0] == "move_down"
    assert KEYMAP["up"][0] == "move_up"
    assert "j" not in KEYMAP
    assert "k" not in KEYMAP
    assert KEYMAP["right"][0] == "forward"
    assert KEYMAP["left"][0] == "back"
    assert KEYMAP["space"][0] == "toggle"
    assert KEYMAP["ctrl+s"][0] == "save"
    assert KEYMAP["/"][0] == "filter"
    assert KEYMAP["?"][0] == "help"
    assert KEYMAP["p"][0] == "pause"
    assert KEYMAP["e"][0] == "edit"
