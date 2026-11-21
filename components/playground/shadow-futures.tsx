"use client";

import { useMemo, useState } from "react";
import { Math as Equation } from "@/components/ui/math";
import { SHADOW_RUNS, SHADOW_STEPS, shadowFutures, type MarketPreset } from "@/lib/model/shelf-market";
import styles from "./comparison-budget.module.css";

type ShadowSetup = {
  market: MarketPreset;
  frame: { goldSales: number; nextPoint: number };
  equalShelf?: boolean;
};

const percent = (value: number) => (100 * value).toFixed(1) + "%";

function FuturePanel({ setup, sales, rotate, seed }: {
  setup: ShadowSetup; sales: number; rotate: boolean; seed: number;
}) {
  const { market, frame, equalShelf } = setup;
  const name = equalShelf ? "Equal shelf space" : market.name;
  const { paths, litProbability } = useMemo(
    () => shadowFutures(seed, sales, frame.goldSales, { market, rotate, equalShelf }),
    [seed, sales, frame.goldSales, market, rotate, equalShelf],
  );
  const litCount = paths.filter(path => path.some(sale => sale.rivalWon)).length;
  return (
    <section className={styles.futurePanel} aria-label={`${name} futures`}>
      <header><h3>{name}</h3><p><strong>{litCount}</strong> of {SHADOW_RUNS} lit</p></header>
      <ol className={styles.futureGrid} aria-label={`${name} sampled paths of ${SHADOW_STEPS} sales`}>
        {paths.map((path, index) => {
          const lit = path.some(sale => sale.rivalWon);
          const outcomes = path.map(sale => sale.winner === 0 ? "Gold" : "Teal").join(", ");
          const label = `Path ${index + 1}. ${lit ? "Rival won at least once" : "Favorite won all five"}. ${outcomes}.`;
          return (
            <li key={index} data-lit={lit} aria-label={label} title={label}>
              {path.map((sale, step) => <i key={step} data-product={sale.winner === 0 ? "gold" : "teal"} aria-hidden="true" />)}
            </li>
          );
        })}
      </ol>
      <dl className={styles.futureOdds}>
        <div><dt>Chance the rival wins next</dt><dd>{percent(frame.nextPoint)}</dd></div>
        <div><dt>Chance of a lit five-sale path</dt><dd>{percent(litProbability)}</dd></div>
      </dl>
    </section>
  );
}

export function ShadowFutures({ first, second, sales, rotate, seed }: {
  first: ShadowSetup; second: ShadowSetup; sales: number; rotate: boolean; seed: number;
}) {
  const [batch, setBatch] = useState(0);
  return (
    <section id="shadow-futures" className={styles.shadows} aria-labelledby="shadows-title">
      <header>
        <h2 id="shadows-title">The histories one shelf never shows</h2>
        <p>From the chart’s current position ({sales.toLocaleString("en-US")} {sales === 1 ? "sale" : "sales"} completed), simulate the next five sales 24 ways in each market. The same simulated customers visit both markets in each pair of strips.</p>
        <p>Each strip is one possible future. Read its five sales from left to right. A strip lights up if the less likely product wins even once. A grey strip means the favorite won every sale.</p>
      </header>
      <div className={styles.futureKey}><span><i data-product="gold" /> Gold sale</span><span><i data-product="teal" /> Teal sale</span><span><i className={styles.greySwatch} /> Favorite won all five</span></div>
      <div className={styles.futurePanels}>
        <FuturePanel setup={first} sales={sales} rotate={rotate} seed={seed + batch} />
        <FuturePanel setup={second} sales={sales} rotate={rotate} seed={seed + batch} />
      </div>
      <button type="button" className={styles.textButton} onClick={() => setBatch(value => value + 1)}>Resample the 24 futures <span aria-hidden="true">↻</span></button>
      <div className={styles.futureMath}>
        <h3>What makes a future light up?</h3>
        <p><Equation block={false} latex={String.raw`\varepsilon_t = 1 - \max_i p_{i,t}`} /> gives the rival’s chance before each sale. At 99–1 odds the sale adds 0.01 points to the chart. At 55–45 it adds 0.45. The chart sums those chances, <Equation block={false} latex={String.raw`B_T=\sum_{t=0}^{T-1}\varepsilon_t`} />. The strips show who actually wins in each sampled future.</p>
        <details>
          <summary>How shelf position changes the odds</summary>
          <p><Equation block={false} latex={String.raw`p_{i,t}\propto e^{\beta x_i}(1+n_{i,t})^{\rho_t}`} />. Here <i>x</i> is product quality, <i>n</i> is past sales, and <i>ρ</i> controls how strongly a lead feeds back into the next sale. At full ranking strength, ρ is {first.market.reinforcement} for {first.market.name} and {second.equalShelf ? 0 : second.market.reinforcement} for {second.equalShelf ? "equal shelf space" : second.market.name}. Rotating the shelf temporarily sets ρ to zero.</p>
          <p>To calculate the five-sale probability, follow the path where the favorite wins every time. Update the odds after each win and multiply those five winning probabilities together. Subtract that product from one to get the chance that a path lights up. Resampling changes the strips while these probabilities stay fixed.</p>
        </details>
      </div>
    </section>
  );
}
