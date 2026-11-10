"""3D cabin view: an original wide-body seat map rendered as SVG, tilted with
CSS perspective. Picking a class flies the camera to that cabin and lights
up the passenger's seat in the verdict colour."""
import hashlib

import streamlit as st


W, H = 1400, 300
TOP, BOT = 78, 222          # inner cabin wall (y)

# x-ranges and seating pattern per cabin; pattern = seats per block, aisles between blocks
CABINS = {
    "Business": {"x": (180, 520), "rows": list(range(11, 20)), "blocks": [1, 2, 1], "letters": "ADGK",
                 "seat": (26, 22), "zoom": 1.95, "color": "#c9a46a"},
    "Eco Plus": {"x": (585, 712), "rows": list(range(30, 34)), "blocks": [2, 4, 2], "letters": "ACDEFGHK",
                 "seat": (20, 13), "zoom": 2.7, "color": "#7fb3d5"},
    "Eco": {"x": (750, 1290), "rows": list(range(40, 49)) + list(range(53, 70)), "blocks": [3, 3, 3],
            "letters": "ABCDEGHJK", "seat": (14, 11.5), "zoom": 1.6, "color": "#5d8a7a"},
}
LABELS = {"Business": "Business", "Eco Plus": "Premium Economy", "Eco": "Economy"}


def _seat_y(blocks, seat_h):
    """Y centres for every seat across the cabin, with aisles between blocks."""
    n_seats, n_aisles = sum(blocks), len(blocks) - 1
    usable = BOT - TOP - 10
    aisle = (usable - n_seats * (seat_h + 2)) / max(n_aisles, 1)
    ys, y = [], TOP + 5
    for b, count in enumerate(blocks):
        for _ in range(count):
            ys.append(y + seat_h / 2)
            y += seat_h + 2
        if b < n_aisles:
            y += aisle
    return ys


def _seat_svg(cls, cfg, chosen):
    x0, x1 = cfg["x"]
    rows = cfg["rows"]
    sw, sh = cfg["seat"]
    ys = _seat_y(cfg["blocks"], sh)
    gap_after = rows.index(48) if cls == "Eco" else None  # mid-cabin galley in economy
    pitch = (x1 - x0 - (30 if gap_after is not None else 0)) / len(rows)
    out = []
    for r_i, row in enumerate(rows):
        x = x0 + r_i * pitch + (30 if gap_after is not None and r_i > gap_after else 0)
        for s_i, y in enumerate(ys):
            seat_id = f"{row}{cfg['letters'][s_i]}"
            is_me = seat_id == chosen
            rot = ""
            if cls == "Business":  # herringbone suites angled toward the aisle
                rot = f' transform="rotate({-28 if s_i in (0, 2) else 28} {x + sw / 2} {y})"'
            out.append(f'<rect class="seat{" me" if is_me else ""}" x="{x:.1f}" y="{y - sh / 2:.1f}" width="{sw - 3:.1f}" '
                       f'height="{sh:.1f}" rx="3"{rot} fill="{cfg["color"]}"/>')
            if is_me:
                out.append(f'<circle class="ping" cx="{x + sw / 2 - 1.5:.1f}" cy="{y:.1f}" r="{max(sw, sh):.1f}"/>')
        if cls != "Eco" or r_i % 3 == 0:
            out.append(f'<text class="rowno" x="{x + sw / 2 - 1:.1f}" y="{TOP - 6}">{row}</text>')
    return "".join(out)


def pick_seat(cls: str, passenger_key: str) -> str:
    """Deterministic seat per passenger so the same inputs land on the same seat."""
    cfg = CABINS[cls]
    h = int(hashlib.md5(passenger_key.encode()).hexdigest(), 16)
    return f"{cfg['rows'][h % len(cfg['rows'])]}{cfg['letters'][(h // 7) % len(cfg['letters'])]}"


def _plane_svg(selected, seat):
    zones, seats = [], []
    for cls, cfg in CABINS.items():
        x0, x1 = cfg["x"]
        dim = "" if cls == selected else " dim"
        zones.append(f'<rect class="zone{dim}" x="{x0 - 14}" y="{TOP}" width="{x1 - x0 + 22}" height="{BOT - TOP}" rx="6" '
                     f'style="--c:{cfg["color"]}"/>')
        seats.append(f'<g class="cabin{dim}">{_seat_svg(cls, cfg, seat if cls == selected else None)}</g>')
    mid = (TOP + BOT) / 2
    return f"""
<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="hull" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#2a3a57"/><stop offset=".5" stop-color="#1a2741"/><stop offset="1" stop-color="#2a3a57"/>
    </linearGradient>
    <linearGradient id="wing" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="#1f2d47"/><stop offset="1" stop-color="#162238"/>
    </linearGradient>
  </defs>
  <!-- wings -->
  <path d="M640 {TOP - 4} L760 {TOP - 4} L930 -60 L880 -60 Z" fill="url(#wing)" stroke="#2c3e5e"/>
  <path d="M640 {BOT + 4} L760 {BOT + 4} L930 {H + 60} L880 {H + 60} Z" fill="url(#wing)" stroke="#2c3e5e"/>
  <!-- tailplanes -->
  <path d="M1250 {TOP + 10} L1320 {TOP + 10} L1385 {TOP - 70} L1355 {TOP - 70} Z" fill="url(#wing)" stroke="#2c3e5e"/>
  <path d="M1250 {BOT - 10} L1320 {BOT - 10} L1385 {BOT + 70} L1355 {BOT + 70} Z" fill="url(#wing)" stroke="#2c3e5e"/>
  <!-- fuselage -->
  <path d="M60 {mid} C 70 {TOP - 2}, 120 {TOP - 14}, 190 {TOP - 14} L1250 {TOP - 14} C 1330 {TOP - 14}, 1390 {mid - 18}, 1395 {mid}
           C 1390 {mid + 18}, 1330 {BOT + 14}, 1250 {BOT + 14} L190 {BOT + 14} C 120 {BOT + 14}, 70 {BOT + 2}, 60 {mid} Z"
        fill="url(#hull)" stroke="#3a5075" stroke-width="2"/>
  <path d="M86 {mid - 14} q 14 -14 30 -16 v 32 q -16 -2 -30 -16 Z" fill="#0b1220" opacity=".85"/>
  <rect x="150" y="{TOP}" width="1150" height="{BOT - TOP}" rx="10" fill="#0f1a2c"/>
  {''.join(zones)}
  <!-- galleys / lavatories -->
  <rect x="534" y="{TOP + 8}" width="34" height="34" rx="4" class="svc"/><rect x="534" y="{BOT - 42}" width="34" height="34" rx="4" class="svc"/>
  <rect x="722" y="{mid - 18}" width="20" height="36" rx="4" class="svc"/>
  <rect x="1004" y="{mid - 22}" width="26" height="44" rx="4" class="svc"/>
  <rect x="158" y="{mid - 20}" width="16" height="40" rx="4" class="svc"/>
  {''.join(seats)}
</svg>"""


def render(selected: str, previous: str, seat: str, verdict_good: bool, p: float, cabin_rates: dict, stage: str):
    def cam(cls, focus=True):
        cx = sum(CABINS[cls]["x"]) / 2 / W * 100
        return (f"rotateX(42deg) scale({CABINS[cls]['zoom']}) translateX({50 - cx:.2f}%)" if focus
                else "rotateX(55deg) scale(1) translateX(0%)")

    start = cam(previous) if previous in CABINS and previous != selected else cam(selected, focus=False)
    end = cam(selected)
    color = "#3ddc97" if verdict_good else "#ff5d5d"
    rate = cabin_rates.get(selected)
    chips = "".join(
        f'<div class="chip{" on" if c == selected else ""}"><b style="color:{CABINS[c]["color"]}">●</b> {LABELS[c]}'
        f'<span>{"{p:.0f}% satisfied".format(p=cabin_rates.get(c, 0) * 100)}</span></div>' for c in CABINS)
    html = f"""
<link href="https://fonts.googleapis.com/css2?family=Big+Shoulders+Display:wght@800&family=JetBrains+Mono:wght@400;600&family=Instrument+Sans:wght@500&display=swap" rel="stylesheet">
<style>
  html,body {{ margin:0; background:transparent; overflow:hidden; font-family:'Instrument Sans',sans-serif; color:#e9eef6; }}
  .wrap {{ position:relative; height:400px; border-radius:14px; overflow:hidden;
           background: radial-gradient(900px 300px at 50% 70%, rgba(90,176,255,.10), transparent 70%),
                       linear-gradient(#0d1628, #0a111f); border:1px solid #22314a; }}
  .wrap::before {{ content:""; position:absolute; inset:0; background:
           repeating-linear-gradient(0deg, rgba(255,255,255,.025) 0 1px, transparent 1px 28px),
           repeating-linear-gradient(90deg, rgba(255,255,255,.025) 0 1px, transparent 1px 28px); }}
  .stage {{ position:absolute; inset:0; perspective:1300px; perspective-origin:50% 30%; }}
  .plane {{ position:absolute; left:2%; right:2%; top:28%; transform-style:preserve-3d;
            transform:{start}; transition:transform 1.5s cubic-bezier(.65,.02,.2,1);
            filter: drop-shadow(0 40px 30px rgba(0,0,0,.55)); }}
  .plane svg {{ width:100%; height:auto; display:block; overflow:visible; }}
  .zone {{ fill:var(--c); opacity:.10; transition:opacity .8s; }}
  .zone.dim {{ opacity:.03; }}
  .cabin.dim {{ opacity:.28; }}
  .seat {{ stroke:#0b1220; stroke-width:1; }}
  .seat.me {{ fill:{color}; stroke:#fff; stroke-width:1.2; }}
  .ping {{ fill:none; stroke:{color}; stroke-width:2; transform-box:fill-box; transform-origin:center;
           animation:ping 1.6s ease-out infinite; }}
  @keyframes ping {{ from {{ transform:scale(.6); opacity:1 }} to {{ transform:scale(2.4); opacity:0 }} }}
  .svc {{ fill:#1e2c45; stroke:#2f4366; }}
  .rowno {{ fill:#6d7f9c; font:600 8px 'JetBrains Mono',monospace; text-anchor:middle; }}
  .hud {{ position:absolute; left:18px; top:14px; z-index:2; max-width:calc(100% - 200px); }}
  @media (max-width: 560px) {{
    .hud {{ max-width:calc(100% - 130px); }} .hud .s {{ display:none; }} .hud .t {{ font-size:1.5rem; }}
    .tag .seatno {{ font-size:1.5rem; }} .chip {{ font-size:.66rem; padding:4px 8px; }}
  }}
  .hud .k {{ font:600 .68rem 'JetBrains Mono',monospace; letter-spacing:.2em; color:#ffb627; }}
  .hud .t {{ font:800 2rem 'Big Shoulders Display',sans-serif; text-transform:uppercase; line-height:1; }}
  .hud .s {{ font:400 .8rem 'JetBrains Mono',monospace; color:#8a9bb4; margin-top:4px; }}
  .tag {{ position:absolute; right:18px; top:14px; z-index:2; text-align:right; }}
  .tag .seatno {{ font:600 1.9rem 'JetBrains Mono',monospace; color:{color}; line-height:1; }}
  .tag .lbl {{ font:400 .7rem 'JetBrains Mono',monospace; letter-spacing:.18em; color:#8a9bb4; }}
  .chips {{ position:absolute; left:14px; right:14px; bottom:12px; display:flex; gap:8px; flex-wrap:wrap; z-index:2; }}
  .chip {{ font:400 .74rem 'JetBrains Mono',monospace; padding:5px 10px; border-radius:99px; background:rgba(18,28,46,.85);
           border:1px solid #22314a; color:#8a9bb4; }}
  .chip.on {{ color:#e9eef6; border-color:#ffb627; }}
  .chip span {{ margin-left:8px; color:#e9eef6; }}
</style>
<div class="wrap" id="wrap">
  <div class="hud"><div class="k">{"CABIN VIEW"} · {stage.upper()}</div><div class="t">{LABELS[selected]}</div>
    <div class="s">{"Hold-out passengers in this cabin: {p:.1f}% predicted satisfied".format(p=rate * 100 if rate is not None else 0)}</div></div>
  <div class="tag"><div class="lbl">{"PASSENGER SEAT"}</div><div class="seatno">{seat}</div>
    <div class="lbl" style="color:{color}">{"P(SATISFIED)"} {p:.2f}</div></div>
  <div class="stage"><div class="plane" id="plane">{_plane_svg(selected, seat)}</div></div>
  <div class="chips">{chips}</div>
</div>
<script>
  const plane = document.getElementById('plane'), wrap = document.getElementById('wrap');
  const end = "{end}";
  requestAnimationFrame(() => setTimeout(() => {{ plane.style.transform = end; }}, 80));
  // subtle parallax: tilt toward the cursor once the fly-in is done
  setTimeout(() => {{
    plane.style.transition = 'transform .35s ease-out';
    wrap.addEventListener('mousemove', e => {{
      const r = wrap.getBoundingClientRect();
      const dx = (e.clientX - r.left) / r.width - .5, dy = (e.clientY - r.top) / r.height - .5;
      plane.style.transform = end + ` rotateY(${{dx * 6}}deg) rotateX(${{-dy * 6}}deg)`;
    }});
    wrap.addEventListener('mouseleave', () => {{ plane.style.transform = end; }});
  }}, 1600);
</script>"""
    # HTML is generated entirely by this module (no user strings), so iframe embedding is safe.
    st.iframe(html, height=404)
