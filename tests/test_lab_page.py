"""Route/asset tests for the Program Lab page (/lab)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api.app import app

ROOT = os.path.join(os.path.dirname(__file__), "..")


class TestLabPage(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_lab_route_serves_page(self):
        res = self.client.get("/lab")
        self.assertEqual(res.status_code, 200)
        body = res.get_data(as_text=True)
        self.assertIn("Program Lab", body)
        self.assertIn("/static/js/program_lab.js", body)
        self.assertIn("/static/css/program_lab.css", body)

    def test_lab_assets_exist_and_are_served(self):
        for path in ("/static/js/program_lab.js", "/static/css/program_lab.css"):
            res = self.client.get(path)
            self.assertEqual(res.status_code, 200, path)

    def test_simulator_page_links_to_lab(self):
        """The two pages cross-link.

        The hrefs are emitted via url_for so the UI stays correct when mounted
        under a URL prefix, so assert against the RESOLVED urls rather than
        hardcoded paths (both routes carry an alias: / + /rtl_cpu, /lab +
        /program_lab, and url_for picks one of each).
        """
        with app.test_request_context():
            from flask import url_for
            lab_url = url_for("program_lab")
            index_url = url_for("index")

        body = self.client.get("/").get_data(as_text=True)
        self.assertIn(f'href="{lab_url}"', body)
        # The lab links back to the simulator.
        lab = self.client.get("/lab").get_data(as_text=True)
        self.assertIn(f'href="{index_url}"', lab)

    def test_page_cross_links_resolve(self):
        """Whatever url_for picked, the emitted links must actually serve."""
        with app.test_request_context():
            from flask import url_for
            urls = [url_for("index"), url_for("program_lab")]
        for url in urls:
            self.assertEqual(self.client.get(url).status_code, 200, url)

    def test_lab_page_ids_match_js_contract(self):
        """Every getElementById in program_lab.js must exist in the template
        (the JS is not null-guarded — it owns this page)."""
        import re
        with open(os.path.join(ROOT, "ui", "static", "js", "program_lab.js")) as f:
            js = f.read()
        with open(os.path.join(ROOT, "ui", "templates", "program_lab.html")) as f:
            html = f.read()
        ids = set(re.findall(r"\$\('([\w-]+)'\)", js))
        self.assertTrue(ids, "expected $('id') lookups in program_lab.js")
        missing = [i for i in sorted(ids) if f'id="{i}"' not in html]
        self.assertEqual(missing, [], f"HTML is missing ids used by the JS: {missing}")


if __name__ == "__main__":
    unittest.main()
