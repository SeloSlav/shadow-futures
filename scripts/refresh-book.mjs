import { existsSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../', import.meta.url));
const bundled = join(homedir(), '.cache', 'codex-runtimes', 'codex-primary-runtime',
  'dependencies', 'python', process.platform === 'win32' ? 'python.exe' : 'bin/python3');
const python = process.env.BOOK_PYTHON || (existsSync(bundled) ? bundled : 'python');
const result = spawnSync(python, ['book/tools/refresh_book.py', ...process.argv.slice(2)], {
  cwd: root, stdio: 'inherit', windowsHide: true,
});
if (result.error) {
  console.error(`Book export could not start: ${result.error.message}. Set BOOK_PYTHON to the Python runtime with the book's dependencies.`);
}
process.exit(result.status ?? 1);
