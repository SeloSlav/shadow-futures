"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import {
  OPEN_SHELF_RATE, ROTATION_LENGTH, ROTATION_START, SHADOW_SALES, TOTAL_SALES,
  shadowSales, shelfHistory,
} from "@/lib/model/shelf-market";
import styles from "./comparison-budget.module.css";

const count = new Intl.NumberFormat("en-US");
const points = new Intl.NumberFormat("en-US", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
const percent = (probability: number) => (100 * probability).toFixed(1) + "%";
const seed = 42;
const yMax = TOTAL_SALES / 2;
const xTicks = [0, 1000, 2000, 3000, TOTAL_SALES];
const yTicks = [0, 500, 1000, 1500, yMax];

function BudgetChart({ history, sales, rotate }: {
  history: ReturnType<typeof shelfHistory>;
  sales: number;
  rotate: boolean;
}) {
  const frame = history[sales];
  const coords = (t: number) => `${t},${(yMax - history[t].budget).toFixed(2)}`;
  const vertices = ["M" + coords(0)];
  for (let t = 25; t < sales; t += 25) vertices.push("L" + coords(t));
  vertices.push("L" + coords(sales));
  const curve = vertices.join(" ");
  return (
    <div className={styles.figure} role="img" aria-label={`After ${count.format(sales)} ${sales === 1 ? "sale" : "sales"} the comparison budget is ${points.format(frame.budget)} points. With equal shelf space throughout it would be ${points.format(sales * OPEN_SHELF_RATE)}.`}>
      <div className={styles.axisTitle}>Comparison points</div>
      <div className={styles.plot}>
        {yTicks.map(tick => (
          <div key={tick} className={styles.gridLine} style={{ bottom: `${100 * tick / yMax}%` }}>
            <span>{count.format(tick)}</span>
          </div>
        ))}
        {rotate && <div className={styles.rotationBand} style={{ left: `${100 * ROTATION_START / TOTAL_SALES}%`, width: `${100 * ROTATION_LENGTH / TOTAL_SALES}%` }} />}
        <svg viewBox={`0 0 ${TOTAL_SALES} ${yMax}`} preserveAspectRatio="none" aria-hidden="true">
          <path className={styles.area} d={`${curve} L${sales},${yMax} Z`} />
          <path className={styles.reference} d={`M0,${yMax} L${sales},${(yMax - sales * OPEN_SHELF_RATE).toFixed(2)}`} />
          <path className={styles.curve} d={curve} />
        </svg>
        <i className={styles.endPoint} style={{ left: `${100 * sales / TOTAL_SALES}%`, bottom: `${(100 * frame.budget / yMax).toFixed(3)}%` }} />
        {sales >= 2_500 && <span className={styles.chartNote} style={{ left: `${100 * sales / TOTAL_SALES}%`, bottom: `${(100 * frame.budget / yMax).toFixed(3)}%` }}>
          +{points.format(frame.budget - history[sales - 1000].budget)} in the last 1,000 sales
        </span>}
        {xTicks.map(tick => <span key={tick} className={styles.xTick} style={{ left: `${100 * tick / TOTAL_SALES}%` }}>{count.format(tick)}</span>)}
      </div>
      <div className={styles.xTitle}>Sales completed</div>
    </div>
  );
}

function ShadowFutures() {
  const [batch, setBatch] = useState(0);
  const markets = useMemo(() => shadowSales(seed + batch), [batch]);
  const goldWins = markets.filter(sales => sales > SHADOW_SALES / 2).length;
  const ties = markets.filter(sales => sales === SHADOW_SALES / 2).length;
  return (
    <section className={styles.shadows} aria-labelledby="shadows-title">
      <div>
        <h2 id="shadows-title">The histories one shelf never shows</h2>
        <p>Give Gold the same small edge in 24 separate markets. This time, past sales count toward shelf position from the first customer. Only the sequence of customer choices changes between markets.</p>
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
  const [rotate, setRotate] = useState(false);
  const [sales, setSales] = useState(TOTAL_SALES);
  const [playFrom, setPlayFrom] = useState<number | null>(null);
  const history = useMemo(() => shelfHistory(seed, { rotate }), [rotate]);
  const frame = history[sales];
  const windowSize = Math.min(sales, 1_000);
  const recentBudget = frame.budget - history[sales - windowSize].budget;

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
    setSales(TOTAL_SALES);
  }

  return (
    <main id="main-content" className={styles.page}>
      <header className={styles.intro}>
        <h1>The comparison budget</h1>
        <p><strong>Gold</strong> is slightly better than <strong>Teal</strong>. Both start at the front of the shelf. As sales move the bestseller into the best spot, the other product gets fewer chances. The comparison budget is the sum of those chances.</p>
      </header>
      <section className={styles.exhibit} aria-labelledby="chart-title">
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
              <span><i />{rotate ? "With a shelf rotation" : "Sales-ranked shelf"}</span>
              <span><i className={styles.dash} />Always equal shelf space</span>
            </div>
            <BudgetChart history={history} sales={sales} rotate={rotate} />
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
              ? "In the shaded interval, sales 3,001–3,300, both products get equal shelf space. The quality edge stays the same. Ranking then resumes."
              : "Both products start with equal shelf space. After 150 sales, past wins gradually begin to buy better placement. The dashed line shows what the budget would be if shelf space stayed equal."}</p>
          </div>
          <aside className={styles.readout} aria-label="Budget at the selected sale">
            <h3>Comparison budget</h3>
            <div className={styles.total}><b>{points.format(frame.budget)}</b> points</div>
            <p className={styles.recent}>{sales > 0
              ? <>The last <strong>{count.format(windowSize)} {windowSize === 1 ? "sale" : "sales"}</strong> added <strong>{points.format(recentBudget)} points</strong>.</>
              : "The first customer has yet to choose."}</p>
            <div className={styles.next}>
              <h3>Chance of the next sale</h3>
              <div className={styles.odds}><span style={{ width: percent(frame.goldOdds) }} /></div>
              <div className={styles.oddsLabels}><span>Gold <b>{percent(frame.goldOdds)}</b></span><span>Teal <b>{percent(1 - frame.goldOdds)}</b></span></div>
              <p>The less likely product has a <strong>{percent(frame.nextPoint)}</strong> chance. The next sale adds <strong>{frame.nextPoint.toFixed(3)} points</strong> to the budget, whoever wins.</p>
            </div>
          </aside>
        </div>
      </section>
      <div className={styles.definition}><span aria-hidden="true">Σ</span><p><strong>How to read one point.</strong> At 90–10 odds, each sale adds 0.10 points. Ten sales at those odds add one point. The budget adds up the chances a rival had each time a customer chose.</p></div>
      <ShadowFutures />
      <footer className={styles.footer}>
        <Link href="/math">Explore the mathematics</Link>
        <a href="https://www.spawn.co/@metalhorse233/the-complexity-science-museum">Visit the original museum exhibit</a>
      </footer>
    </main>
  );
}
