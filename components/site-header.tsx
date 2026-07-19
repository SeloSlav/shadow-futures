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
          <Link href="/#breakout">Start the story</Link>
          <Link href="/methodology">Methodology</Link>
          <Link href="/math">Mathematics</Link>
        </nav>
      </div>
    </header>
  );
}
