import type { Metadata } from "next";
import type { ReactNode } from "react";

import "@/app/globals.css";
import { SiteHeader } from "@/components/site-header";
import { BOOK_DESCRIPTION, BOOK_FULL_TITLE } from "@/lib/book";
import { AUTHOR } from "@/lib/paper/citation";
import { SITE_ORIGIN } from "@/lib/site";

export const metadata: Metadata = {
  metadataBase: new URL(SITE_ORIGIN),
  title: {
    default: "Shadow Futures by Martin Erlic",
    template: "%s | Shadow Futures",
  },
  description: BOOK_DESCRIPTION,
  authors: [{ name: AUTHOR.name, url: AUTHOR.path }],
  creator: AUTHOR.name,
  alternates: { canonical: "/" },
  icons: {
    icon: [
      { url: "/favicon.ico", sizes: "any" },
      { url: "/favicon-48.png", type: "image/png", sizes: "48x48" },
    ],
    apple: [{ url: "/apple-touch-icon.png", type: "image/png", sizes: "180x180" }],
  },
  verification: {
    google: "mDWlpJpnFhzdXa9kacEFsEEvw52h2BBhZud3XVHo24g",
  },
  openGraph: {
    type: "website",
    title: BOOK_FULL_TITLE,
    description: BOOK_DESCRIPTION,
    siteName: "Shadow Futures",
    url: "/",
  },
  twitter: {
    card: "summary_large_image",
    title: BOOK_FULL_TITLE,
    description: BOOK_DESCRIPTION,
  },
};

export default function RootLayout({ children }: Readonly<{ children: ReactNode }>) {
  return (
    <html lang="en">
      <body>
        <a className="skip-link" href="#main-content">
          Skip to content
        </a>
        <SiteHeader />
        {children}
      </body>
    </html>
  );
}
