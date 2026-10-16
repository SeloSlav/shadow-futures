import edition from "@/public/book/edition.json";

export const BOOK = edition;
export const BOOK_PDF_SIZE = `${(BOOK.pdf_bytes / 1024 / 1024).toFixed(1)} MB`;
