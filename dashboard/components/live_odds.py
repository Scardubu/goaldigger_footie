"""Live odds component using SSE via a simple HTML embed.
Avoids a Node build by leveraging st.components.v1.html.
"""
from __future__ import annotations

from typing import Optional

import streamlit as st
from streamlit.components.v1 import html as st_html
from dashboard.components.unified_design_system import get_unified_design_system

_DEF_CSS = """
<style>
  .gd-odds { font-family: var(--gd-font-family, system-ui, sans-serif); }
  .gd-odds .row { display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; padding: 6px 8px; }
  .gd-odds .head { font-weight: 600; color: var(--gd-gray-600, #4b5563); }
  .gd-odds .cell { text-align: center; font-variant-numeric: tabular-nums; }
  .gd-odds .market { text-align: left; }
  .gd-odds .up { color: var(--gd-success, #10b981); }
  .gd-odds .down { color: var(--gd-error, #ef4444); }
  .gd-odds .card { border: 1px solid var(--gd-gray-200, #e5e7eb); border-radius: 12px; padding: 8px; background: var(--gd-white, #fff); }
  .gd-odds .title { font-weight: 600; margin: 4px 0 8px; }
</style>
"""


_DEF_HTML = """
<div class="gd-odds card" role="region" aria-live="polite" aria-label="Live odds">
  <div class="title">Live Odds</div>
  <div class="row head">
    <div>Market</div>
    <div class="cell">Home</div>
    <div class="cell">Draw</div>
    <div class="cell">Away</div>
  </div>
  <div id="gd-rows"></div>
</div>
<script>
(function() {
  const url = %(url_json)s;
  const rows = document.getElementById('gd-rows');
  const fmt = (n) => Number(n).toFixed(2);
  const state = new Map();
  function render(data){
    // Key by market
    const key = data.market || '1X2';
    const prev = state.get(key);
    state.set(key, data);
    const changed = prev ? (data.home > prev.home ? 'up' : (data.home < prev.home ? 'down' : '')) : '';
    let el = document.getElementById('row-' + key);
    const html = `
      <div class="row" id="row-${key}">
        <div class="market">${key}</div>
        <div class="cell ${changed}">${fmt(data.home)} ${changed==='up'?'▲':changed==='down'?'▼':''}</div>
        <div class="cell">${fmt(data.draw)}</div>
        <div class="cell">${fmt(data.away)}</div>
      </div>`;
    if (!el) {
      const container = document.createElement('div');
      container.innerHTML = html;
      rows.appendChild(container.firstElementChild);
    } else {
      el.outerHTML = html;
    }
  }
  try {
    const es = new EventSource(url);
    es.onmessage = function(ev) {
      try { const data = JSON.parse(ev.data); if (data && data.type === 'odds') render(data); } catch (e) {}
    };
    es.onerror = function() { /* ignore for demo */ };
  } catch (e) { /* ignore */ }
})();
</script>
"""


def render_live_odds_sse(url: str = "http://localhost:8079/events", height: int = 240) -> None:
    """Render a simple live odds viewer bound to an SSE endpoint.
    Args:
        url: SSE endpoint URL
        height: component iframe height
    """
  # Prepare markup
  markup = _DEF_CSS + _DEF_HTML % {"url_json": repr(url)}

  # Try to render inside UnifiedDesignSystem card for consistent styling
  try:
    uds = get_unified_design_system()
    uds.inject_unified_css(dashboard_type="premium")

    def _odds_embed():
      # Using the combined markup inside the card body
      st_html(markup, height=height)

    uds.create_unified_card(_odds_embed)
  except Exception:
    # Fallback to raw embed
    st_html(markup, height=height)
