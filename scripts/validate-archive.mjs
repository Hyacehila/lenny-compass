import { createHash } from 'node:crypto';
import { access, readdir, readFile, stat } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));
const repositoryRoot = path.resolve(scriptDirectory, '..');
const archiveRoot = path.join(repositoryRoot, 'archive');
const indexPath = path.join(archiveRoot, '01-start-here', 'index.json');
const manifestPath = path.join(repositoryRoot, 'docs', 'archive-manifest.json');

const toPortablePath = value => value.split(path.sep).join('/');
const sha256 = async filePath => createHash('sha256').update(await readFile(filePath)).digest('hex');

const listFiles = async directory => {
  const entries = await readdir(directory, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const entryPath = path.join(directory, entry.name);
    if (entry.isDirectory()) files.push(...await listFiles(entryPath));
    if (entry.isFile()) files.push(entryPath);
  }
  return files.sort((left, right) => left.localeCompare(right));
};

const requireFile = async filePath => {
  try {
    await access(filePath);
  } catch {
    throw new Error(`Missing required file: ${toPortablePath(path.relative(repositoryRoot, filePath))}`);
  }
};

await requireFile(indexPath);
await requireFile(path.join(archiveRoot, '01-start-here', 'README.md'));
await requireFile(path.join(archiveRoot, '01-start-here', 'LICENSE.md'));

const index = JSON.parse(await readFile(indexPath, 'utf8'));
const newsletters = Array.isArray(index.newsletters) ? index.newsletters : [];
const podcasts = Array.isArray(index.podcasts) ? index.podcasts : [];
if (newsletters.length !== 349 || podcasts.length !== 289) {
  throw new Error(`Expected 349 newsletters and 289 podcasts, received ${newsletters.length} and ${podcasts.length}.`);
}

const indexedItems = [...newsletters, ...podcasts];
for (const item of indexedItems) {
  if (!item || typeof item.filename !== 'string' || !item.filename.endsWith('.md')) {
    throw new Error('Every indexed item must expose a Markdown filename.');
  }
  const resolvedPath = path.resolve(archiveRoot, item.filename);
  if (!resolvedPath.startsWith(`${archiveRoot}${path.sep}`)) {
    throw new Error(`Unsafe indexed path: ${item.filename}`);
  }
  await requireFile(resolvedPath);
}

const files = await listFiles(archiveRoot);
let totalBytes = 0;
for (const filePath of files) totalBytes += (await stat(filePath)).size;

await requireFile(manifestPath);
const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
const manifestFiles = Array.isArray(manifest?.archive?.files) ? manifest.archive.files : [];
if (manifest.archive.fileCount !== files.length || manifest.archive.totalBytes !== totalBytes) {
  throw new Error('Archive file count or total bytes do not match the manifest. Regenerate the manifest after an intentional source update.');
}

const byPath = new Map(manifestFiles.map(file => [file.path, file]));
if (byPath.size !== files.length) {
  throw new Error('Archive manifest file list does not match the archive file count.');
}

for (const filePath of files) {
  const relativePath = toPortablePath(path.relative(repositoryRoot, filePath));
  const expected = byPath.get(relativePath);
  if (!expected) throw new Error(`Archive manifest is missing ${relativePath}.`);
  const actualStat = await stat(filePath);
  if (expected.bytes !== actualStat.size || expected.sha256 !== await sha256(filePath)) {
    throw new Error(`Archive checksum mismatch: ${relativePath}`);
  }
}

const indexSha256 = await sha256(indexPath);
if (manifest.archive.index.sha256 !== indexSha256) {
  throw new Error('Index checksum does not match the manifest.');
}

console.log(JSON.stringify({
  valid: true,
  schemaVersion: index.schema_version,
  indexGeneratedAt: index.generated_at,
  newsletters: newsletters.length,
  podcasts: podcasts.length,
  indexedItems: indexedItems.length,
  files: files.length,
  totalBytes
}, null, 2));
