import pytest

from services.shopping import SEARCH_TOOLS, ShoppingWorker


def _record_direct(worker, item_number):
    index = item_number - 1
    worker._cart_click_pending = index
    return worker._record_item_added({
        "item_number": item_number,
        "product": f"Product {item_number}",
        "quantity": "1 pack",
        "price": "$2.00",
    })


def test_batches_pause_for_substitution_decisions():
    worker = ShoppingWorker()
    worker._plan_shopping_list([f"item {number}" for number in range(1, 7)])

    assert _record_direct(worker, 1)["status"] == "next_item"
    assert _record_direct(worker, 2)["status"] == "next_item"
    worker._current_item_search_performed = True
    assert worker._record_item_unavailable({
        "item_number": 3,
        "closest_match": {
            "product": "Closest Product",
            "item_type": "same type",
            "quantity": "1 pack",
            "price": "$2.10",
        },
    })["status"] == "next_item"
    assert _record_direct(worker, 4)["status"] == "next_item"
    result = _record_direct(worker, 5)

    assert result == {"status": "review_required", "batch_number": 1, "item_numbers": [3]}
    assert worker._resolve_batch_review("continue")["status"] == "needs_decision"
    assert worker._shopping_items[2]["status"] == "unavailable"

    decision = worker._resolve_batch_review("use the closest match for item 3")
    assert decision["status"] == "resolved"
    assert worker._shopping_items[2]["status"] == "substitute_approved"
    assert _record_direct(worker, 3)["status"] == "next_batch"
    assert _record_direct(worker, 6)["status"] == "all_done"

    summary = worker._final_summary()
    assert "✅ Product 1" in summary
    assert "Approved substitute: Product 3" in summary
    assert "✅ Product 3" not in summary
    assert "Not added:" not in summary


def test_unavailable_item_requires_a_search_and_final_summary_marks_only_direct_matches():
    worker = ShoppingWorker()
    worker._plan_shopping_list(["milk", "bread"])
    with pytest.raises(ValueError, match="Search FairPrice"):
        worker._record_item_unavailable({
            "item_number": 1,
            "closest_match": {"product": "", "item_type": "", "quantity": "", "price": ""},
        })

    worker._current_item_search_performed = True
    assert worker._record_item_unavailable({
        "item_number": 1,
        "closest_match": {"product": "", "item_type": "", "quantity": "", "price": ""},
    })["status"] == "next_item"
    assert _record_direct(worker, 2)["status"] == "review_required"
    assert worker._resolve_batch_review("skip item 1")["status"] == "resolved"

    summary = worker._final_summary()
    assert "✅ Product 2" in summary
    assert "milk — skipped at your request" in summary
    assert "✅ milk" not in summary


def test_progress_text_tracks_searches_and_confirmed_cart_additions():
    worker = ShoppingWorker()
    worker._plan_shopping_list(["oat milk"])
    assert "Shopping list received" in worker._progress_message(
        "plan_shopping_list", {}, {"status": "list_registered", "batch_size": 1, "total_items": 1}
    )
    assert "Searching FairPrice for item 1: oat milk" in worker._progress_message(
        "search_current_item", {}, {"status": "search_submitted", "item_number": 1}
    )

    worker._cart_click_pending = 0
    result = worker._record_item_added({
        "item_number": 1,
        "product": "Oat Drink",
        "quantity": "1 L",
        "price": "$3.50",
    })
    assert "✅ Added item 1 to the cart: Oat Drink" in worker._progress_message(
        "record_item_added", {"item_number": 1}, result
    )


def test_item_must_be_searched_before_unavailable_can_be_recorded():
    class Browser:
        def search_products(self, query):
            assert query == "oat milk"
            return {"url": "https://www.fairprice.com.sg/search?query=oat%20milk", "title": "Results for oat milk"}

    worker = ShoppingWorker()
    worker._browser = Browser()
    worker._plan_shopping_list(["oat milk"])
    search_tool_names = {tool["function"]["name"] for tool in SEARCH_TOOLS}
    assert "add_product_to_cart" not in search_tool_names
    assert "record_item_added" not in search_tool_names

    with pytest.raises(ValueError, match="Search FairPrice"):
        worker._record_item_unavailable({
            "item_number": 1,
            "closest_match": {"product": "", "item_type": "", "quantity": "", "price": ""},
        })

    result = worker._execute("search_current_item", {"item_number": 1}, None, set())
    assert result["status"] == "search_submitted"
    assert worker._current_item_search_performed is True
