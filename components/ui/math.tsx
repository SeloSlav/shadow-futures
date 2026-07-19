"use client";

import katex from "katex";
import { useMemo } from "react";

type MathProps = {
  latex: string;
  block?: boolean;
  label?: string;
  className?: string;
};

export function Math({ latex, block = true, label, className = "" }: MathProps) {
  const html = useMemo(
    () =>
      katex.renderToString(latex, {
        displayMode: block,
        throwOnError: false,
        strict: false,
        output: "htmlAndMathml",
      }),
    [block, latex],
  );

  const classes = `equation ${block ? "equation--block" : "equation--inline"} ${className}`;

  if (!block) {
    return (
      <span
        className={classes}
        aria-label={label}
        dangerouslySetInnerHTML={{ __html: html }}
      />
    );
  }

  return (
    <div
      className={classes}
      aria-label={label}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
