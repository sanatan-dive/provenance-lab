"""Browser end-to-end test of the web app (headless Chromium). Server must be running on :8000.

For each content type: open the tab, load a real held-out sample, click Examine, wait for the verdict,
check the report rendered, and screenshot it. Also screenshots the results page and a phone-width view.
    .venv/bin/python tools/e2e_ui.py
Screenshots go to results/screens/.
"""
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path("results/screens")
OUT.mkdir(parents=True, exist_ok=True)
errors = []

with sync_playwright() as p:
    b = p.chromium.launch()
    page = b.new_page(viewport={"width": 1360, "height": 860}, device_scale_factor=2)
    page.on("console", lambda m: m.type == "error" and errors.append(m.text))
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto("http://127.0.0.1:8000/", wait_until="networkidle")
    page.add_style_tag(content="html{scroll-behavior:auto!important}")
    page.screenshot(path=OUT / "01_hero.png")

    for i, (kind, label) in enumerate([("text", 1), ("code", 0), ("image", 1), ("audio", 0)]):
        page.click(f'.kinds [data-kind="{kind}"]')
        page.click(f'.sample[data-label="{label}"]')
        page.wait_for_function("document.getElementById('hint').textContent.startsWith('Loaded')")
        page.click("#go")
        page.wait_for_selector("#filled:not([hidden])", timeout=120000)
        page.wait_for_function("!document.getElementById('go').disabled", timeout=120000)
        page.wait_for_timeout(1200)  # let the needle settle
        verdict = page.inner_text("#verdict")
        truth = page.inner_text("#truth")
        print(f"{kind}: {verdict} | {truth}")
        page.locator(".bench").screenshot(path=OUT / f"0{i + 2}_{kind}.png")
        page.evaluate("document.getElementById('hint').textContent=''")

    for sel, name in [("#cards", "10_cards"), ("#baselines", "11_baselines"), ("#market", "12_market"),
                      ("#ood", "13_ood"), ("#compare", "14_compare"), ("#curves", "15_curves"), ("#robust", "16_robust")]:
        loc = page.locator(sel)
        if loc.count() and loc.inner_html().strip():
            loc.screenshot(path=OUT / f"{name}.png")
    page.screenshot(path=OUT / "20_full.png", full_page=True)

    m = b.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=2)
    m.goto("http://127.0.0.1:8000/", wait_until="networkidle")
    m.screenshot(path=OUT / "30_mobile.png", full_page=False)
    b.close()

print("console errors:", errors or "none")
