# UI Redesign: Analytical Trading Dashboard

## Overview

Complete visual and structural overhaul of the Tennis Predictor web dashboard. Replaces the current vertical card list layout with a Bloomberg-style trading dashboard featuring a sidebar match list and detail panel. Stays within the existing Python static site generator architecture.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Visual direction | Analytical Terminal | Data-first, professional, information-dense. Bloomberg-meets-betting-exchange. |
| Accent color | Orange `#f97316` | Aggressive, confident. Trading/finance energy. Green reserved for positive signals. |
| Architecture | Keep Python generator | No framework migration. The visual ceiling is the same — the upgrade is CSS/JS, not tooling. Fits existing CI/CD. |
| Layout | Trading Dashboard (sidebar + detail) | Everything on one screen. No scrolling through cards. Click match → see full analysis. |
| Mobile strategy | Two-state slide transition | Match list and detail as two full-screen states with GPU-accelerated slide animations. |

## Visual Identity

### Color System

```
Dark Theme (default):
  --bg:           #050505    Page background
  --surface:      #080808    Elevated surface (sidebar, panels)
  --card:         #0a0a0a    Match items, detail sections
  --card-hover:   #0e0e0e    Hover state
  --border:       #111111    Borders and dividers
  --border-hover: #1a1a1a    Hover border
  --t:            #e5e5e5    Primary text
  --t2:           #888888    Secondary text
  --t3:           #555555    Muted text
  --t4:           #333333    Dim text / subtle elements
  --orange:       #f97316    Primary accent
  --orange-dim:   rgba(249,115,22,0.1)   Orange tint background
  --green:        #22c55e    Positive (winner, +EV, high confidence)
  --green-dim:    rgba(34,197,94,0.08)   Green tint background
  --red:          #ef4444    Negative (loss, error)
  --red-dim:      rgba(239,68,68,0.08)   Red tint background
  --amber:        #eab308    Warning / key factors
  --amber-dim:    rgba(234,179,8,0.08)   Amber tint background

Light Theme:
  --bg:           #f5f5f5
  --surface:      #ffffff
  --card:         #ffffff
  --card-hover:   #fafafa
  --border:       #e5e5e5
  --border-hover: #d0d0d0
  --t:            #1a1a1a
  --t2:           #555555
  --t3:           #888888
  --t4:           #aaaaaa
  --orange:       #ea580c
  --orange-dim:   rgba(234,88,12,0.08)   Orange tint background
  --green:        #16a34a
  --green-dim:    rgba(22,163,74,0.06)   Green tint background
  --red:          #dc2626
  --red-dim:      rgba(220,38,38,0.06)   Red tint background
  --amber:        #d97706
  --amber-dim:    rgba(217,119,6,0.06)   Amber tint background
```

### Typography

- **UI text**: Inter (Google Fonts), weights 400-900
- **Data/numbers**: JetBrains Mono (Google Fonts), weights 400-700
- **Scale**: 8px (labels) → 9px (meta) → 11px (body) → 12-14px (names) → 16-22px (metrics) → 36px (detail probability)
- **Letter spacing**: Tight (-0.5px to -2px) on large numbers, wide (1-2px) on uppercase labels

### Spacing & Radius

- Sidebar width: 280px (desktop), 220px (tablet)
- Card padding: 10-16px
- Section gaps: 6-12px
- Border radius: 4px (badges) → 8px (items) → 10px (panels) → 14-16px (major containers)

## Layout Architecture

### Desktop (>960px)

```
┌─────────────────────────────────────────────────────────┐
│ TOPBAR: Brand | Nav Tabs (Predictions/Results/Analytics) | Theme │
├──────────┬──────────────────────────────────────────────┤
│ SIDEBAR  │ TOPBAR-2: Match title | Metrics strip        │
│          ├──────────────────────────────────────────────┤
│ Match    │                                              │
│ List     │  DETAIL PANEL                                │
│          │                                              │
│ grouped  │  ┌─────────────┐  ┌────────────────────────┐ │
│ by       │  │  Face-off   │  │  H2H + Key Factors     │ │
│ tourney  │  │  + Stat     │  │                        │ │
│          │  │  Bars       │  │                        │ │
│          │  └─────────────┘  └────────────────────────┘ │
│          │                                              │
├──────────┴──────────────────────────────────────────────┤
│ FOOTER                                                  │
└─────────────────────────────────────────────────────────┘
```

- Sidebar is fixed-position, scrollable independently
- Detail panel is fluid width
- Clicking a match in the sidebar swaps detail content with crossfade
- Metrics strip always visible above detail content

### Tablet (769-960px)

Same structure, sidebar narrows to 220px. Detail panel compresses proportionally.

### Small Tablet (481-768px)

```
┌─────────────────────────────────┐
│ TOPBAR                          │
├─────────────────────────────────┤
│ MATCH STRIP (horizontal scroll) │
│ [Alcaraz 73%] [Sinner 81%] ... │
├─────────────────────────────────┤
│                                 │
│ DETAIL PANEL (full width)       │
│                                 │
└─────────────────────────────────┘
```

- Sidebar collapses to horizontal scrollable strip of compact chips
- Each chip: favorite name abbreviated (e.g. "Alcaraz") + probability + confidence dot. No underdog name (too cramped).
- Active chip: orange-dim background + orange border. Inactive: card background + border.
- Strip shows ~3 chips visible before scroll. `scroll-snap-type: x mandatory` for clean snapping.
- When the active chip scrolls out of view, strip auto-scrolls to center it (`scrollIntoView({inline:'center', behavior:'smooth'})`)
- Detail panel fills full width below
- Tapping a chip loads detail with slide-up transition: `translateY(20px) + opacity:0 → translateY(0) + opacity:1`, 200ms ease. This is listed in the animation table.

### Mobile (<480px)

```
STATE 1: Match List               STATE 2: Detail View
┌───────────────────────┐         ┌───────────────────────┐
│ TOPBAR: Brand         │         │ TOPBAR: ← Alc vs Med  │
├───────────────────────┤         ├───────────────────────┤
│                       │  tap →  │                       │
│ MIAMI OPEN            │ ──────► │ Face-off + Prob       │
│ ● Alcaraz vs Med  73% │ ◄────── │ Stat bars             │
│ ● Sinner vs Rub   81% │  ← back │ H2H                   │
│ ● Djokovic vs Fri 64% │         │ Key factors           │
│                       │         │                       │
│ MONTE CARLO           │         │                       │
│ ● Tsitsipas vs R  62% │         │                       │
│                       │         │                       │
├───────────────────────┤         ├───────────────────────┤
│ [Pred] [Results] [Stats]│       │ [Pred] [Results] [Stats]│
└───────────────────────┘         └───────────────────────┘
```

- Two full-screen states driven by a single JS state machine
- Transitions use `transform: translateX()` (GPU-accelerated, 60fps)
- Bottom fixed tab nav replaces top nav
- Swipe-right gesture on detail returns to list

## Components

### 1. Topbar

- Sticky, glass-blur background (`backdrop-filter: blur(20px)`)
- Left: Brand icon (orange gradient square with "T") + "Tennis Predictor" + "PRO" tag
- Center/Right: Nav tabs (pill buttons in a segmented control) + theme toggle
- Active tab: orange-dim background + orange text
- Mobile: simplified brand only, nav moves to bottom

### 2. Metrics Strip

- 4-5 stat cells in a dense grid row, always visible
- Each cell: uppercase label (8px, monospace) + large value (16-22px, JetBrains Mono, bold) + sub-text
- Values color-coded: primary metric = orange, positive = green, neutral = white
- Sits between topbar and detail panel on desktop; below topbar on mobile
- Stats: Accuracy, Brier Score, Grand Slams, High Confidence, Match Count

### 3. Sidebar Match List (desktop/tablet)

- Grouped by tournament with section headers (orange left bar + tournament name)
- Each match item: confidence dot (green/orange/gray with glow) + player names + meta (round if available, surface, tier) + probability (monospace, orange)
- Active item: orange-dim background + orange border
- Hover: subtle background brighten + border appear (150ms)
- Scrolls independently from detail panel

### 4. Detail Panel — Face-off

- Two player columns with initials avatars (56px circles)
- Favorite player: orange border on avatar, orange name
- Center: large probability (36px, JetBrains Mono) + "WIN PROB" label
- Below: player ranks and recent form

### 5. Detail Panel — Stat Comparison Bars

Horizontal dual-bar system (not text rows):

```
        Elo Rating
 2089 ████████████░░░░░ 1987
      Surface Elo
 2145 ██████████████░░░ 1923
      Serve Rating
 1890 █████████░░░░░░░░ 1812
      Return Rating
 1845 ░░░░░░░░░████████ 1878
```

- Each stat: centered label, left bar (favorite), right bar (underdog)
- Bar width proportional to value relative to the pair
- Better value highlighted in orange; worse in gray
- Staggered entrance animation (400ms per bar, 50ms stagger between rows)

Stats shown (display label → JSON field):

| Display Label | Favorite Field | Underdog Field |
|---------------|----------------|----------------|
| Elo | `detail.p1.elo` | `detail.p2.elo` |
| Surface Elo | `detail.p1.surface_elo` | `detail.p2.surface_elo` |
| Serve | `detail.p1.serve_elo` | `detail.p2.serve_elo` |
| Return | `detail.p1.return_elo` | `detail.p2.return_elo` |
| 1st Serve % | `detail.p1.first_serve_pct` | `detail.p2.first_serve_pct` |
| 1st Serve Won | `detail.p1.first_serve_won` | `detail.p2.first_serve_won` |
| BP Save % | `detail.p1.bp_save_pct` | `detail.p2.bp_save_pct` |
| Return Pts Won | `detail.p1.return_pts_won` | `detail.p2.return_pts_won` |
| Form (5) | `detail.p1.form_last5` | `detail.p2.form_last5` |
| Surface W/L | `detail.p1.surface_record` | `detail.p2.surface_record` |

Note: `p1`/`p2` map to `player1`/`player2` — the JS determines which is favorite based on `prob_p1`. Percentage fields (values 0-1) are displayed as `Math.round(v * 100) + '%'`. Null/undefined fields are rendered as "—" and the stat row is dimmed.

### 6. Detail Panel — H2H

- Circular indicator: wins count inside a ring, ring partially filled with orange proportional to win ratio
- Text summary: "Alcaraz leads 6-3 overall" + surface-specific record
- Compact, sits in the right column next to stat bars

### 7. Detail Panel — Key Factors

- List with icon badges (single letter in orange-dim square): E (Elo), F (Form), H (H2H), S (Serve), etc.
- Each factor: 1-line description
- Sits below H2H in right column

### 8. Filter Bar

- Row of pill-shaped toggle buttons above the match list
- Filters: All | High Conf | Medium | Hard | Clay | Grass
- Active: orange-dim background + orange border
- Match count badge at the end
- Filters apply to sidebar on desktop, match list on mobile
- Filtering triggers staggered fadeIn re-render (50ms per item)

### 9. Results Tab

- Grouped by day with date headers
- Each day header: date + accuracy badge (green if >65%, orange if >55%, red otherwise)
- Result rows: win/loss circle icon (green checkmark / red X) + predicted winner name + probability + opponent
- Running streak counter at the top ("Current: 4W" or "Current: 2L")

### 10. Analytics Tab

- Upgraded calibration chart (canvas):
  - Animated curve drawing on tab switch (800ms)
  - Hover tooltips showing bin center, actual rate, sample count
  - Dashed diagonal for perfect calibration
  - Second overlay line (dimmed) for bookmaker baseline comparison — uses hardcoded reference curve `[[0.1,0.1],[0.3,0.3],[0.5,0.5],[0.7,0.7],[0.9,0.9]]` (perfect calibration proxy) if bookmaker data is unavailable in `calibration`. Future: pass actual bookmaker calibration from evaluation pipeline.
  - Theme-aware colors
- System info cards in a 2x2 grid (features count, match count, zero leakage, autonomous)

### 11. Player Modal

- Triggered by clicking player name (distinct underline affordance — dotted underline, underline-offset 3px)
- Content: name, rank, ratings section, form section, serve/return section
- Future: radar chart for multi-dimensional view (serve/return/form/surface/mental)
- Desktop: centered overlay with backdrop blur (`rgba(0,0,0,0.7)` + `backdrop-filter:blur(4px)`)
- Mobile: slides up as bottom sheet, max-height 85vh, border-radius on top corners only
- Dismiss: tap backdrop, X button (top-right), or swipe-down on mobile (threshold: 100px downward drag)
- Mobile modal overlays the bottom nav (z-index above it)
- `body` scroll locked while modal is open (`overflow:hidden`)

### 12. Micro-Animations

| Element | Animation | Duration | Easing |
|---------|-----------|----------|--------|
| Probability bars | Width from 0 on first render | 800ms | cubic-bezier(0.4, 0, 0.2, 1) |
| Detail panel swap (desktop) | Crossfade content | 200ms | ease |
| Detail slide-up (small tablet) | translateY(20px)+opacity:0 → translateY(0)+opacity:1 | 200ms | ease |
| Detail slide-in (mobile) | translateX(100%) → translateX(0) | 300ms | cubic-bezier(0.4, 0, 0.2, 1) |
| Detail slide-out (mobile) | translateX(0) → translateX(100%) | 300ms | cubic-bezier(0.4, 0, 0.2, 1) |
| Stat bars (detail) | Width stagger on load | 400ms each, 50ms stagger | ease-out |
| Match item hover | Background + border brighten | 150ms | ease |
| Match item tap (mobile) | Background pulse | 100ms | ease |
| Tab switch | View opacity fade | 150ms | ease |
| Filter re-render | Staggered fadeIn per match item | 50ms stagger | ease |
| High-conf dot | Subtle pulse glow | 2s infinite | ease-in-out |
| Calibration curve | Stroke draw | 800ms | ease-out |
| Bottom nav active | Scale press (0.95) | 100ms | ease |
| Skeleton loading | Shimmer gradient sweep | 1.5s infinite | linear |

All animations respect `prefers-reduced-motion: reduce` — disabled when set.

## Mobile Transition Detail

### State Machine

```
State: { view: 'list' | 'detail', selectedMatch: string | null, tab: 'predictions' | 'results' | 'analytics' }
```

- Single state object drives both desktop and mobile rendering
- `matchMedia('(max-width: 480px)')` listener determines transition style
- Resizing from desktop → mobile mid-session gracefully snaps to correct state (see Orientation & Resize below)

### Slide Transition Implementation

Both states (list and detail) are sibling containers:

```
.mobile-container {
  position: relative;
  overflow: hidden;
}
.mobile-list, .mobile-detail {
  position: absolute;
  inset: 0;
  transition: transform 300ms cubic-bezier(0.4, 0, 0.2, 1);
  will-change: transform;
}
```

- **List → Detail**: list slides to `translateX(-30%)` (parallax), detail slides from `translateX(100%)` to `translateX(0)`
- **Detail → List**: reverse. Detail slides to `translateX(100%)`, list slides from `translateX(-30%)` to `translateX(0)`
- Parallax offset on the list (-30% instead of -100%) creates depth, matching iOS navigation patterns
- `will-change: transform` ensures composited layer for 60fps
- Touch: `touchstart`/`touchmove`/`touchend` on detail view tracks horizontal swipe (see Swipe Gesture Detail below)

### Bottom Tab Nav

- Fixed at bottom, 56px height
- 3 tabs with icon + label
- Active tab: orange color + scale feedback on press
- `body` gets `padding-bottom: 56px` to prevent content occlusion
- Tabs work identically to desktop nav tabs — switch between Predictions/Results/Analytics views

### Browser Back Button (History API)

Entering detail view pushes a history entry: `history.pushState({view:'detail', match: id}, '')`. The `popstate` listener triggers the back-to-list transition. This means:

- User taps match → detail slides in, history entry pushed
- User presses browser back → `popstate` fires → list slides back in
- User presses back again → normal browser navigation (leaves page)
- Desktop: no history manipulation (sidebar is always visible, no "back" concept)

### Swipe Gesture Detail

Touch swipe-right on the detail view to return to match list:

- `touchstart`: record start X/Y position and timestamp
- `touchmove`: if horizontal distance > 10px and angle < 30° from horizontal, the detail panel follows the finger (`transform: translateX(deltaX)`) with the list parallax appearing behind (`translateX(-30% + deltaX*0.3)`)
- `touchend`:
  - If distance > 80px OR velocity > 0.5px/ms: trigger back transition (animate to final positions)
  - Else: snap back to detail view (animate to `translateX(0)`)
- Velocity = `distance / (endTime - startTime)`
- During drag, apply `transition: none` to prevent fighting. Re-enable transition on release.

### Orientation & Resize

The `matchMedia` listener fires on orientation change and window resize:

- **Phone landscape (>480px width)**: switches to small-tablet strip layout. If user was in mobile detail state, the detail panel content is preserved and displayed below the strip. The match chip auto-activates in the strip.
- **Phone portrait (<480px)**: switches to two-state mobile. If a match was selected in strip layout, snaps directly to detail state (no animation on layout switch — only on user-initiated transitions).
- **Tablet → Desktop**: sidebar appears, detail panel fills. No animation on resize, just CSS breakpoint snap.
- Resize transitions use `transition: none` temporarily to prevent janky intermediate states, re-enabled after 100ms debounce.

### Touch Targets

- Match list items: full-width, min 52px height
- Player name links: min 44px tap area (padding extended beyond visible text)
- Filter pills: min 36px height
- Bottom nav buttons: full flex-1 width, 56px height
- Back button in detail view: 44x44px

## Technical Architecture

### File Output

```
site/
├── index.html          # Single HTML shell with all structure
├── predictions.json    # Data payload (regenerated by Python)
├── sw.js              # Service worker
└── assets/
    ├── css/style.css  # All styles (~600-800 lines)
    └── js/app.js      # All JavaScript (~800-1200 lines)
```

### Generator Structure

`src/tennis_predictor/web/generate.py` remains the sole generator. Internal functions:

- `_write_html()` — HTML shell with sidebar, detail panel, results, analytics, modal, mobile containers
- `_write_css()` — Complete CSS with theme system, responsive breakpoints, animations
- `_write_js()` — State machine, rendering functions, chart drawing, event handlers, touch gestures
- `_write_sw()` — Service worker (unchanged)

### JS State Management

```javascript
const state = {
  view: 'list',           // 'list' | 'detail' (mobile only)
  tab: 'predictions',     // 'predictions' | 'results' | 'analytics'
  selectedMatch: null,     // match ID
  filters: { conf: 'all', surface: 'all' },
  isMobile: false          // set by matchMedia listener
};
```

All rendering is driven by `state`. Changing `selectedMatch` triggers detail panel render (crossfade on desktop, slide on mobile). Changing `tab` triggers view swap. Changing `filters` triggers match list re-render with stagger animation.

### Data Contract

No changes to `predictions.json` structure. The JS reads the existing fields:

- `predictions[].player1, player2, prob_p1, p1_rank, p2_rank`
- `predictions[].confidence_tier, surface, tournament, model` — note: `confidence_tier` is optional (only present when selective model enrichment runs). JS defaults to `''` if missing and hides confidence badge.
- `predictions[].round` — optional, may not be present on all predictions. Displayed if available, omitted gracefully if not.
- `predictions[].detail.p1, detail.p2` (player stats objects, see field mapping in Component 5)
- `predictions[].detail.h2h` — `{total, p1_wins, p2_wins}`. Entire section hidden if `h2h.total` is falsy.
- `predictions[].detail.factors` — array of strings. Section hidden if empty array.
- `model_stats` — `{accuracy, brier_score, n_matches}`. Falls back to hardcoded defaults if missing.
- `calibration` — `{bin_centers, actual_rates}`. Chart shows "No data" message if missing.
- `history` — array of `{date, count, predictions}`. Results tab shows empty state if missing.

### Initial / Empty / Loading States

| State | Behavior |
|-------|----------|
| Page load (before fetch) | Skeleton shimmer in sidebar (6 placeholder items) + detail panel (placeholder bars). Metrics strip shows "—" values. |
| Fetch failed | Sidebar: "Could not load predictions" with retry button. Detail panel: empty state with error message. |
| No predictions available | Sidebar: "No matches scheduled" message. Detail: instructional text "Check back during tournament weeks." |
| No match selected (desktop initial) | Detail panel shows: model summary card (accuracy, Brier, features count) + "Select a match from the sidebar" prompt. Auto-selects the first match after 1s if user hasn't clicked. |
| No match selected (mobile) | N/A — mobile starts in list state, detail is only reachable by tapping a match. |
| No history data | Results tab: "No prediction history yet. Results appear after predictions are tracked against outcomes." |
| No calibration data | Analytics tab: chart area shows "Calibration data not yet available" centered text instead of canvas. |

## What's NOT Changing

- Python generator pattern (no React/Vite/Next.js)
- `predictions.json` data contract
- GitHub Pages deployment
- Service worker
- CI/CD workflows
- Data pipeline or model code
