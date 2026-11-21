"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import {
  MARKET_PRESETS, ROTATION_LENGTH, ROTATION_START, SHADOW_SALES, TOTAL_SALES,
  goldOddsAtFullRanking, shadowSales, shelfHistory, type MarketPreset,
} from "@/lib/model/shelf-market";
import styles from "./comparison-budget.module.css";

const count = new Intl.NumberFormat("en-US");
const points = new Intl.NumberFormat("en-US", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
const percent = (probability: number) => (100 * probability).toFixed(1) + "%";
const seed = 42;
const yMax = TOTAL_SALES / 2;
const xTicks = [0, 1000, 2000, 3000, TOTAL_SALES];
const yTicks = [0, 500, 1000, 1500, yMax];

function budgetLine(name: string, history: ReturnType<typeof shelfHistory>, sales: number, equalShelf = false) {
  const frame = history[equalShelf ? 0 : sales];
  const budgetAt = (t: number) => equalShelf ? t * frame.nextPoint : history[t].budget;
  const coords = (t: number) => `${t},${(yMax - budgetAt(t)).toFixed(2)}`;
  const vertices = ["M" + coords(0)];
  if (!equalShelf) {
    for (let t = 25; t < sales; t += 25) vertices.push("L" + coords(t));
  }
  vertices.push("L" + coords(sales));
  return {
    name, path: vertices.join(" "), budget: budgetAt(sales),
    recent: budgetAt(sales) - budgetAt(Math.max(0, sales - 1000)),
    goldOdds: frame.goldOdds, nextPoint: frame.nextPoint,
  };
}

type BudgetLine = ReturnType<typeof budgetLine>;

function BudgetChart({ lines, sales, rotate }: { lines: BudgetLine[]; sales: number; rotate: boolean }) {
  const totals = lines.map(line => `${line.name} has ${points.format(line.budget)} points`).join(" and ");
  return (
    <div className={styles.figure} role="img" aria-label={`After ${count.format(sales)} ${sales === 1 ? "sale" : "sales"}, ${totals}.`}>
      <div className={styles.axisTitle}>Comparison points</div>
      <div className={styles.plot}>
        {yTicks.map(tick => (
          <div key={tick} className={styles.gridLine} style={{ bottom: `${100 * tick / yMax}%` }}>
            <span>{count.format(tick)}</span>
          </div>
        ))}
        {rotate && <div className={styles.rotationBand} style={{ left: `${100 * ROTATION_START / TOTAL_SALES}%`, width: `${100 * ROTATION_LENGTH / TOTAL_SALES}%` }} />}
        <svg viewBox={`0 0 ${TOTAL_SALES} ${yMax}`} preserveAspectRatio="none" aria-hidden="true">
          {lines.map(line => <path key={line.name} className={styles.curve} data-series={line.name} d={line.path} />)}
        </svg>
        {lines.map((line, i) => <i key={line.name} className={styles.endPoint} data-comparison={i === 1}
          style={{ left: `${100 * sales / TOTAL_SALES}%`, bottom: `${(100 * line.budget / yMax).toFixed(3)}%` }} />)}
        {xTicks.map(tick => <span key={tick} className={styles.xTick} style={{ left: `${100 * tick / TOTAL_SALES}%` }}>{count.format(tick)}</span>)}
      </div>
      <div className={styles.xTitle}>Sales completed</div>
    </div>
  );
}

function MarketAssumptions({ market }: { market: MarketPreset }) {
  return <p><strong>{market.name}.</strong> {market.description} A 20–10 sales lead at full ranking strength gives Gold a {percent(goldOddsAtFullRanking(market, 20, 10))} chance of the next sale.</p>;
}

function ShadowFutures({ market }: { market: MarketPreset }) {
  const [batch, setBatch] = useState(0);
  const markets = useMemo(() => shadowSales(seed + batch, market), [batch, market]);
  const goldWins = markets.filter(sales => sales > SHADOW_SALES / 2).length;
  const ties = markets.filter(sales => sales === SHADOW_SALES / 2).length;
  return (
    <section className={styles.shadows} aria-labelledby="shadows-title">
      <div>
        <h2 id="shadows-title">The histories one shelf never shows</h2>
        <p>Run the <strong>{market.name}</strong> setup 24 times with the same two rivals. Here, shelf ranking starts at full strength with the first customer. Only the sequence of customer choices changes between runs.</p>
        <button type="button" className={styles.textButton} onClick={() => setBatch(value => value + 1)}>Rerun the 24 markets <span aria-hidden="true">↻</span></button>
      </div>
      <div className={styles.markets}>
        <h3>Who sold more after 1,000 sales?</h3>
        <div className={styles.marketGrid} role="img" aria-label={`Twenty-four simulated outcomes. Gold leads in ${goldWins}, Teal in ${markets.length - goldWins - ties}, and ${ties} are tied.`}>
          {markets.map((sales, i) => (
            <div key={i} className={styles.market} title={`Market ${i + 1} · Gold ${sales}, Teal ${SHADOW_SALES - sales}`}>
              <span style={{ width: percent(sales / SHADOW_SALES) }} />
            </div>
          ))}
        </div>
        <p className={styles.winners} aria-live="polite"><span><i className={styles.goldDot} /> Gold leads in <b>{goldWins}</b></span><span><i className={styles.tealDot} /> Teal leads in <b>{markets.length - goldWins - ties}</b></span>{ties > 0 && <span>{ties} tied</span>}</p>
        <p className={styles.note}>Each bar shows how one market’s sales split between the two products.</p>
      </div>
    </section>
  );
}

export function ComparisonBudget() {
  const [market, setMarket] = useState(MARKET_PRESETS[0]);
  const [comparison, setComparison] = useState<MarketPreset | null>(() => MARKET_PRESETS.find(preset => preset.name === "Commodities") ?? null);
  const [rotate, setRotate] = useState(false);
  const [sales, setSales] = useState(TOTAL_SALES);
  const [playFrom, setPlayFrom] = useState<number | null>(null);
  const history = useMemo(() => shelfHistory(seed, { market, rotate }), [market, rotate]);
  const otherHistory = useMemo(() => comparison && shelfHistory(seed, { market: comparison, rotate }), [comparison, rotate]);
  const firstLine = budgetLine(market.name, history, sales);
  const otherLine = comparison && otherHistory
    ? budgetLine(comparison.name, otherHistory, sales)
    : budgetLine("Equal shelf space", history, sales, true);
  const lines = [firstLine, otherLine];
  const gap = Math.abs(firstLine.budget - otherLine.budget);
  const larger = firstLine.budget >= otherLine.budget ? firstLine.name : otherLine.name;
  const windowSize = Math.min(sales, 1_000);

  useEffect(() => {
    if (playFrom === null) return;
    const start = performance.now();
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const id = window.setInterval(() => {
      const t = Math.min(TOTAL_SALES, playFrom + Math.floor((performance.now() - start) / 2));
      setSales(t);
      if (t === TOTAL_SALES) setPlayFrom(null);
    }, reducedMotion ? 1_000 : 50);
    return () => window.clearInterval(id);
  }, [playFrom]);

  function play() {
    if (playFrom !== null) { setPlayFrom(null); return; }
    const from = sales === TOTAL_SALES ? 0 : sales;
    setSales(from);
    setPlayFrom(from);
  }

  function selectShelf(rotated: boolean) {
    if (rotated === rotate) return;
    setPlayFrom(null);
    setRotate(rotated);
  }

  return (
    <main id="main-content" className={styles.page}>
      <header className={styles.intro}>
        <h1>Comparison budget playground</h1>
        <p>Compare two markets over the same 4,000 sales. In each, <strong>Gold</strong> has a small quality edge over <strong>Teal</strong>. Each sale adds the less likely rival’s chance of winning to that market’s comparison budget.</p>
      </header>
      <section className={styles.exhibit} aria-labelledby="chart-title">
        <div className={styles.marketSetup}>
          <div className={styles.pickers}>
            <label>First market
              <select value={market.name} onChange={e => {
                const selected = MARKET_PRESETS.find(preset => preset.name === e.target.value);
                if (!selected) return;
                setPlayFrom(null);
                if (selected === comparison) setComparison(market);
                setMarket(selected);
              }}>
                {MARKET_PRESETS.map(preset => <option key={preset.name}>{preset.name}</option>)}
              </select>
            </label>
            <label>Compare with
              <select value={comparison?.name ?? ""} onChange={e => {
                setPlayFrom(null);
                setComparison(MARKET_PRESETS.find(preset => preset.name === e.target.value) ?? null);
              }}>
                {MARKET_PRESETS.map(preset => <option key={preset.name} disabled={preset === market}>{preset.name}</option>)}
                <optgroup label="Benchmark"><option value="">Equal shelf space</option></optgroup>
              </select>
            </label>
          </div>
          <details className={styles.assumptions}>
            <summary>About these setups</summary>
            <MarketAssumptions market={market} />
            {comparison ? <MarketAssumptions market={comparison} /> : <p><strong>Equal shelf space.</strong> The two products keep the {market.name} quality difference. Past sales have no effect on placement.</p>}
          </details>
        </div>
        <div className={styles.exhibitHead}>
          <h2 id="chart-title">How much comparison survives?</h2>
          <div className={styles.switcher} role="group" aria-label="Shelf rule">
            <button type="button" aria-pressed={!rotate} onClick={() => selectShelf(false)}>Rank by sales</button>
            <button type="button" aria-pressed={rotate} onClick={() => selectShelf(true)}>Rotate the shelf</button>
          </div>
        </div>
        <div className={styles.body}>
          <div className={styles.chartColumn}>
            <div className={styles.legend}>
              {lines.map(line => <div key={line.name}><span><i />{line.name}</span><strong>{points.format(line.budget)} <small>pts</small></strong></div>)}
            </div>
            <BudgetChart lines={lines} sales={sales} rotate={rotate} />
            <div className={styles.transport}>
              <button type="button" className={styles.play} onClick={play}>
                <span aria-hidden="true">{playFrom !== null ? "Ⅱ" : "▶"}</span>
                {playFrom !== null ? "Pause" : sales === TOTAL_SALES ? "Replay sales" : "Play sales"}
              </button>
              <input aria-label="Sales completed" type="range" min={0} max={TOTAL_SALES} value={sales}
                aria-valuetext={`${count.format(sales)} sales completed`}
                onChange={e => { setPlayFrom(null); setSales(Number(e.target.value)); }} />
              <span>{count.format(sales)} {sales === 1 ? "sale" : "sales"}</span>
            </div>
            <p className={styles.caption}>{rotate
              ? "In the shaded interval, sales 3,001–3,300, both products get equal shelf space in each ranked market. Ranking then resumes."
              : "The products start on equal shelves. After 150 sales, past wins gradually begin to buy better placement in each ranked market."}</p>
          </div>
          <aside className={styles.readout} aria-label="Budget at the selected sale">
            <h3>Budget gap</h3>
            <div className={styles.total}><b>{points.format(gap)}</b> points</div>
            <p className={styles.recent}>{gap < 0.05 ? "Less than 0.05 points separate the two setups." : <>By sale {count.format(sales)}, the <strong>{larger}</strong> setup has given rivals more chances.</>}</p>
            <div className={styles.recentGains}>
              <h3>{windowSize ? `Last ${count.format(windowSize)} ${windowSize === 1 ? "sale" : "sales"}` : "Before the first sale"}</h3>
              <dl>{lines.map(line => <div key={line.name}><dt>{line.name}</dt><dd>+{points.format(line.recent)} pts</dd></div>)}</dl>
            </div>
            <details className={styles.next}>
              <summary>Chance of the next sale</summary>
              {lines.map(line => <div key={line.name}>
                <h4>{line.name}</h4>
                <div className={styles.odds}><span style={{ width: percent(line.goldOdds) }} /></div>
                <div className={styles.oddsLabels}><span>Gold <b>{percent(line.goldOdds)}</b></span><span>Teal <b>{percent(1 - line.goldOdds)}</b></span></div>
                <p>Next sale adds <strong>{line.nextPoint.toFixed(3)} points</strong>.</p>
              </div>)}
            </details>
          </aside>
        </div>
      </section>
      <div className={styles.definition}><span aria-hidden="true">Σ</span><p><strong>How to read one point.</strong> At 90–10 odds, each sale adds 0.10 points. Ten sales at those odds add one point. The budget adds up the chances a rival had each time a customer chose.</p></div>
      <ShadowFutures market={market} />
      <footer className={styles.footer}>
        <Link href="/math">Explore the mathematics</Link>
      </footer>
    </main>
  );
}
