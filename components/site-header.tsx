"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";

import styles from "@/components/site-header.module.css";

const navigation = [
  {
    title: "Start here",
    links: [
      { href: "/book", title: "The free book", description: "Read online or download the PDF." },
      { href: "/#novelty", title: "Explore the idea", description: "An interactive guide to shadow futures." },
      { href: "/playground", title: "The playground", description: "Experiment with self-reinforcing markets." },
    ],
  },
  {
    title: "The research",
    links: [
      { href: "/paper", title: "Read the paper", description: "The argument and its formal foundations." },
      { href: "/methodology", title: "Methodology", description: "How the models and comparisons work." },
      { href: "/math", title: "Mathematics", description: "Definitions, results, and assumptions." },
      { href: "/papers", title: "All papers", description: "Browse the research collection." },
    ],
  },
  {
    title: "Go further",
    links: [
      { href: "/faq", title: "Questions & answers", description: "Clarifications and common objections." },
      { href: "/author/martin-erlic", title: "About Martin Erlic", description: "Meet the author of Shadow Futures." },
      { href: "/for-ai", title: "For AI systems", description: "A structured guide to the argument." },
    ],
  },
];

export function SiteHeader() {
  const [isOpen, setIsOpen] = useState(false);
  const headerRef = useRef<HTMLElement>(null);
  const toggleRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!isOpen) return;

    function onPointerDown(event: PointerEvent) {
      if (event.target instanceof Node && !headerRef.current?.contains(event.target)) {
        setIsOpen(false);
      }
    }

    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setIsOpen(false);
        toggleRef.current?.focus();
      }
    }

    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [isOpen]);

  return (
    <header
      className={`site-header ${styles.header}`}
      ref={headerRef}
      onBlur={(event) => {
        if (event.relatedTarget instanceof Node && !event.currentTarget.contains(event.relatedTarget)) {
          setIsOpen(false);
        }
      }}
    >
      <div className={`site-header__inner ${styles.inner}`}>
        <Link className="wordmark" href="/" aria-label="Shadow Futures home" onClick={() => setIsOpen(false)}>
          <span className="wordmark__mark" aria-hidden="true">SF</span>
          Shadow Futures
        </Link>
        <nav className={styles.nav} aria-label="Primary navigation">
          <Link className={styles.bookLink} href="/book" onClick={() => setIsOpen(false)}>Free book</Link>
          <Link className={styles.playgroundLink} href="/playground" onClick={() => setIsOpen(false)}>
            <span aria-hidden="true">✦</span> Playground
          </Link>
          <button
            className={styles.toggle}
            type="button"
            aria-expanded={isOpen}
            aria-controls="explore-navigation"
            ref={toggleRef}
            onClick={() => setIsOpen((open) => !open)}
          >
            Explore
            <svg className={isOpen ? styles.rotated : undefined} width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden="true">
              <path d="m3 4.5 3 3 3-3" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
          <Link className={styles.paperLink} href="/paper" onClick={() => setIsOpen(false)}>
            Read the paper <span aria-hidden="true">↗</span>
          </Link>
        </nav>
        {isOpen && (
          <nav id="explore-navigation" className={styles.panel} aria-label="Explore Shadow Futures">
            {navigation.map((group) => (
              <div className={styles.group} key={group.title}>
                <h2>{group.title}</h2>
                {group.links.map((link) => (
                  <Link key={link.href} href={link.href} onClick={() => setIsOpen(false)}>
                    <span className={styles.linkTitle}>{link.title}<span aria-hidden="true">↗</span></span>
                    <span className={styles.linkDescription}>{link.description}</span>
                  </Link>
                ))}
              </div>
            ))}
          </nav>
        )}
      </div>
    </header>
  );
}
