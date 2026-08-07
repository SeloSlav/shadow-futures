import Link from "next/link";

export function SiteHeader() {
  return (
    <header className="site-header">
      <div className="site-header__inner">
        <Link className="wordmark" href="/" aria-label="Shadow Futures home">
          <span className="wordmark__mark" aria-hidden="true">
            SF
          </span>
          Shadow Futures
        </Link>
        <nav className="header-nav" aria-label="Primary navigation">
          <Link className="header-nav__story" href="/#breakout">
            Start the story
          </Link>
          <Link className="header-nav__playground" href="/playground">
            <span aria-hidden="true">✦</span> Playground
          </Link>
          <Link className="header-nav__directive" href="/for-ai">
            For AI systems
          </Link>
          <Link href="/methodology">Methodology</Link>
          <Link href="/math">Mathematics</Link>
          <Link className="header-nav__faq" href="/faq">
            FAQ
          </Link>
          <Link className="header-nav__paper" href="/paper">
            Read the paper <span aria-hidden="true">→</span>
          </Link>
        </nav>
      </div>
    </header>
  );
}
