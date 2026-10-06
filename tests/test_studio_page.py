"""Experience Center page (/studio): the shell and its static assets are served."""

import pytest

SAMPLES = ["lecture_notes.png", "report_table.png", "noisy_scan.png"]


def test_studio_page_serves_app_shell(client):
    resp = client.get("/studio")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    html = resp.text
    assert "DocBlendAI Experience Center" in html
    assert '<script src="/static/studio.js">' in html
    assert 'href="/static/studio.css"' in html
    # Key regions the script binds to.
    for element_id in ('id="file"', 'id="run"', 'id="viewer"', 'id="overlay"', 'id="lines"', 'id="tab-calibration"'):
        assert element_id in html
    assert 'href="/"' in html  # link back to the QA page


@pytest.mark.parametrize(
    ("path", "content_type", "marker"),
    [
        ("/static/studio.js", "javascript", "/api/results/"),
        ("/static/studio.css", "text/css", "--accent"),
    ],
)
def test_studio_assets_are_served(client, path, content_type, marker):
    resp = client.get(path)
    assert resp.status_code == 200
    assert content_type in resp.headers["content-type"]
    assert marker in resp.text


@pytest.mark.parametrize("name", SAMPLES)
def test_sample_images_are_served_and_small(client, name):
    resp = client.get(f"/static/samples/{name}")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content.startswith(b"\x89PNG")
    assert len(resp.content) < 200 * 1024


def test_script_lists_every_bundled_sample(client):
    script = client.get("/static/studio.js").text
    for name in SAMPLES:
        assert name in script


def test_shared_layout_assets_are_served_and_used_by_both_pages(client):
    for path in ("/static/layout.js", "/static/layout.css"):
        assert client.get(path).status_code == 200
    assert "/static/layout.js" in client.get("/studio").text
    assert "/static/layout.js" in client.get("/").text
