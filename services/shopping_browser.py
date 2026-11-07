"""A visible Playwright browser for the FairPrice shopping demo.

This module owns the browser session, page observations, and basic page controls
used by the FairPrice shopping agent.
"""
import re
from urllib.parse import urlparse

from playwright.sync_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    TimeoutError as PlaywrightTimeoutError,
    sync_playwright,
)


FAIRPRICE_URL = "https://www.fairprice.com.sg/"
FAIRPRICE_HOST = "www.fairprice.com.sg"
BLOCKED_ACTION = re.compile(r"\b(log\s*in|sign\s*in|sign\s*up|account|check\s*out|checkout|payment|place\s+order|purchase|buy\s+now|pay\s+now)\b", re.I)
BLOCKED_FIELD = re.compile(r"\b(password|passcode|credential|email|payment|credit\s*card|card\s*number|expiry|security\s*code|cvv|cvc)\b", re.I)
BLOCKED_PATH = re.compile(r"/(?:login|sign[-_]?in|sign[-_]?up|checkout|payment|orders?)(?:/|$)", re.I)


class FairPriceBrowser:
    """Launch and control a Chromium session on the FairPrice website."""

    def __init__(self, *, headless: bool = False, slow_mo_ms: int = 0):
        self.headless = headless
        self.slow_mo_ms = slow_mo_ms
        self._playwright: Playwright | None = None
        self.browser: Browser | None = None
        self.context: BrowserContext | None = None
        self.page: Page | None = None
        self.postal_code: str | None = None
        self._delivery_addresses: list[dict[str, str]] = []
        self._element_refs: dict[str, dict] = {}

    def start(self) -> "FairPriceBrowser":
        """Open Chromium and navigate to FairPrice's home page."""
        if self.page is not None:
            return self
        self._playwright = sync_playwright().start()
        try:
            self.browser = self._playwright.chromium.launch(
                headless=self.headless,
                slow_mo=self.slow_mo_ms,
            )
            self.context = self.browser.new_context(
                viewport={"width": 1440, "height": 1000},
            )
            self.page = self.context.new_page()
            self.navigate(FAIRPRICE_URL)
        except Exception:
            self.close()
            raise
        return self

    def navigate(self, url: str) -> dict:
        """Navigate to a FairPrice URL and return basic page metadata."""
        parsed = urlparse(url)
        if (
            parsed.scheme != "https"
            or parsed.hostname != FAIRPRICE_HOST
            or parsed.username
            or parsed.password
            or BLOCKED_PATH.search(parsed.path)
        ):
            raise ValueError(f"Only https://{FAIRPRICE_HOST} pages are supported.")
        page = self._require_page()
        response = page.goto(url, wait_until="domcontentloaded", timeout=60_000)
        return {
            "url": page.url,
            "title": page.title(),
            "status": response.status if response else None,
        }

    def screenshot(self) -> bytes:
        """Capture the current page as PNG bytes for a vision-capable model."""
        return self._require_page().screenshot(type="png", full_page=False)

    def search_products(self, query: str) -> dict:
        """Search FairPrice through its visible product search field."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Provide a product description to search for.")
        page = self._require_page()
        search = page.locator("#search-input-bar")
        search.wait_for(state="visible", timeout=10_000)
        search.fill(query.strip(), timeout=10_000)
        search.press("Enter", timeout=10_000)
        page.wait_for_timeout(1000)
        self._element_refs = {}
        return {"url": page.url, "title": page.title()}

    def page_description(self) -> str:
        """Return visible page text for models that cannot inspect screenshots."""
        page = self._require_page()
        return page.locator("body").inner_text(timeout=15_000)

    def observe(self) -> dict:
        """Read page text and label visible controls for grounded browser actions."""
        page = self._require_page()
        controls = page.evaluate("""() => {
          const selector = 'button, a[href], input:not([type=hidden]), textarea, select, [role=button], [role=option]';
          const visible = (el) => {
            const rect = el.getBoundingClientRect();
            const style = getComputedStyle(el);
            return rect.width > 0 && rect.height > 0 && style.visibility !== 'hidden' && style.display !== 'none';
          };
          return Array.from(document.querySelectorAll(selector)).filter(visible).slice(0, 160).map((el, i) => {
            const ref = `e${i}`;
            el.setAttribute('data-openweights-ref', ref);
            const label = el.getAttribute('aria-label') || el.innerText || el.value || el.placeholder || el.title || el.name || '';
            const addButton = /\\badd\\s+to\\s+cart\\b/i.test(label);
            let context = '';
            if (addButton) {
              let parent = el.parentElement;
              for (let depth = 0; parent && depth < 7; depth++, parent = parent.parentElement) {
                const text = (parent.innerText || '').trim().replace(/\\s+/g, ' ');
                if (text.length >= 24 && text.length <= 900) { context = text; break; }
              }
            }
            return {
              ref,
              tag: el.tagName.toLowerCase(),
              id: el.id || '',
              testid: el.getAttribute('data-testid') || '',
              role: el.getAttribute('role') || (el.tagName === 'A' ? 'link' : el.tagName === 'BUTTON' ? 'button' : el.tagName.toLowerCase()),
              name: label.trim().replace(/\\s+/g, ' ').slice(0, 180),
              context,
              placeholder: el.placeholder || '',
              input_type: el.type || '',
              disabled: Boolean(el.disabled),
              href: el.href || ''
            };
          });
        }""")
        self._element_refs = {control["ref"]: control for control in controls}
        return {
            "url": page.url,
            "title": page.title(),
            "text": self.page_description()[:12_000],
            "controls": controls,
        }

    def click_element(self, ref: str, *, allow_add_to_cart: bool = False) -> None:
        control = self._get_control(ref)
        if control["disabled"]:
            raise ValueError("That page control is disabled.")
        if BLOCKED_ACTION.search(control["name"]):
            raise ValueError("Login and checkout actions are not allowed in this demo.")
        is_add_to_cart = bool(re.search(r"\badd\s+to\s+cart\b", control["name"], re.I))
        if is_add_to_cart and not allow_add_to_cart:
            raise ValueError("Use the item-tracked add_product_to_cart action for product additions.")
        if allow_add_to_cart and not is_add_to_cart:
            raise ValueError("That control is not an Add to cart button.")
        if control["href"]:
            destination = urlparse(control["href"])
            if destination.hostname != FAIRPRICE_HOST or BLOCKED_PATH.search(destination.path):
                raise ValueError("Only FairPrice shopping pages can be opened in this demo.")
        self._require_page().locator(f'[data-openweights-ref="{ref}"]').click(timeout=15_000)
        self._element_refs = {}

    @property
    def delivery_location_open(self) -> bool:
        return self._require_page().locator('input[data-testid="addressSearch"]').is_visible()

    def fill_element(self, ref: str, value: str) -> None:
        control = self._get_control(ref)
        label = " ".join((control["name"], control["placeholder"], control["input_type"]))
        if control["input_type"].lower() == "password" or BLOCKED_FIELD.search(label):
            raise ValueError("Login and payment fields are not allowed in this demo.")
        if control["tag"] not in {"input", "textarea"}:
            raise ValueError("Choose a text field to enter text.")
        self._require_page().locator(f'[data-openweights-ref="{ref}"]').fill(value, timeout=15_000)
        self._element_refs = {}

    def press_key(self, key: str) -> None:
        allowed = {"Enter", "Escape", "Tab", "ArrowDown", "ArrowUp"}
        if key not in allowed:
            raise ValueError(f"Supported keys: {', '.join(sorted(allowed))}.")
        self._require_page().keyboard.press(key)
        self._element_refs = {}

    def scroll_page(self, direction: str, amount: int = 700) -> None:
        if direction not in {"up", "down"}:
            raise ValueError("Scroll direction must be 'up' or 'down'.")
        if not 100 <= amount <= 1200:
            raise ValueError("Scroll amount must be from 100 to 1200 pixels.")
        self._require_page().mouse.wheel(0, amount if direction == "down" else -amount)
        self._element_refs = {}

    def wait(self, milliseconds: int = 1000) -> None:
        if not 100 <= milliseconds <= 5000:
            raise ValueError("Wait must be from 100 to 5000 milliseconds.")
        self._require_page().wait_for_timeout(milliseconds)

    def _get_control(self, ref: str) -> dict:
        control = self._element_refs.get(ref)
        if control is None:
            raise ValueError("That page reference has expired. Observe the page again first.")
        return control

    def control_info(self, ref: str) -> dict:
        """Return the latest observed metadata for a visible control reference."""
        return dict(self._get_control(ref))

    def request_delivery_location(self) -> dict:
        """Open FairPrice's location picker and ask the caller for a postal code."""
        page = self._require_page()
        postal_input = page.locator('input[data-testid="addressSearch"]')
        if not postal_input.is_visible():
            page.get_by_text("Enter your address or postal code", exact=True).first.click(timeout=10_000)
            postal_input.wait_for(state="visible", timeout=10_000)
        return {"status": "postal_code_required"}

    def set_postal_code(self, postal_code: str) -> dict:
        """Look up a postal code, auto-select one result, or return choices to confirm."""
        if not re.fullmatch(r"\d{6}", postal_code):
            raise ValueError("Enter a six-digit Singapore postal code.")
        page = self._require_page()
        postal_input = page.locator('input[data-testid="addressSearch"]')
        if not postal_input.is_visible():
            self.request_delivery_location()
        choices = page.locator('[data-testid="address-selection"]')
        if choices.count() and choices.first.is_visible():
            postal_input.fill("")
            try:
                choices.first.wait_for(state="hidden", timeout=5_000)
            except PlaywrightTimeoutError:
                pass
        postal_input.fill(postal_code)
        try:
            choices.first.wait_for(state="visible", timeout=10_000)
        except PlaywrightTimeoutError:
            self._delivery_addresses = []
            return {"status": "no_address_found", "postal_code": postal_code, "options": []}

        addresses = []
        for choice in choices.all():
            addresses.append({
                "name": choice.locator(".main-text").inner_text().strip(),
                "address": choice.locator(".secondary-text").inner_text().strip(),
            })
        self.postal_code = postal_code
        self._delivery_addresses = addresses

        if len(addresses) == 1:
            return self.select_delivery_address(0)
        if len(addresses) > 1:
            return {
                "status": "address_choice_required",
                "postal_code": postal_code,
                "options": addresses,
            }
        return {"status": "no_address_found", "postal_code": postal_code, "options": []}

    def select_delivery_address(self, index: int) -> dict:
        """Select an address after the caller resolves any multiple-choice prompt."""
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(self._delivery_addresses):
            raise ValueError("Choose one of the address options returned for the postal code.")
        self._require_page().locator('[data-testid="address-selection"]').nth(index).click(timeout=10_000)
        address = self._delivery_addresses[index]
        self._delivery_addresses = []
        self._element_refs = {}
        return {
            "status": "address_selected",
            "postal_code": self.postal_code,
            "address": address,
        }

    def click(self, selector: str) -> None:
        self._require_page().locator(selector).click(timeout=15_000)

    def fill(self, selector: str, value: str) -> None:
        self._require_page().locator(selector).fill(value, timeout=15_000)

    def press(self, selector: str, key: str) -> None:
        self._require_page().locator(selector).press(key, timeout=15_000)

    def scroll(self, amount: int) -> None:
        self._require_page().mouse.wheel(0, amount)

    def go_back(self) -> dict | None:
        page = self._require_page()
        response = page.go_back(wait_until="domcontentloaded", timeout=30_000)
        return {
            "url": page.url,
            "title": page.title(),
            "status": response.status if response else None,
        }

    def close(self) -> None:
        """Close the browser and release the Playwright driver."""
        if self.browser is not None:
            self.browser.close()
        self.browser = None
        self.context = None
        self.page = None
        self.postal_code = None
        self._delivery_addresses = []
        if self._playwright is not None:
            self._playwright.stop()
        self._playwright = None

    def __enter__(self) -> "FairPriceBrowser":
        return self.start()

    def __exit__(self, *_exc) -> None:
        self.close()

    def _require_page(self) -> Page:
        if self.page is None:
            raise RuntimeError("The FairPrice browser is not running. Call start() first.")
        return self.page
