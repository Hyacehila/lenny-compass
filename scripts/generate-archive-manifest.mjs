import { createHash } from 'node:crypto';
import { readdir, readFile, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));
const repositoryRoot = path.resolve(scriptDirectory, '..');
const archiveRoot = path.join(repositoryRoot, 'archive');
const indexPath = path.join(archiveRoot, '01-start-here', 'index.json');
const outputPath = path.join(repositoryRoot, 'docs', 'archive-manifest.json');

const toPortablePath = value => value.split(path.sep).join('/');
const sha256 = async filePath => createHash('sha256').update(await readFile(filePath)).digest('hex');

const listFiles = async directory => {
  const entries = await readdir(directory, { withFileTypes: true });
  const files = [];
  for (const entry of entries.sort((left, right) => left.name.localeCompare(right.name))) {
    const entryPath = path.join(directory, entry.name);
    if (entry.isDirectory()) files.push(...await listFiles(entryPath));
    if (entry.isFile()) files.push(entryPath);
  }
  return files;
};

const index = JSON.parse(await readFile(indexPath, 'utf8'));
const newsletters = Array.isArray(index.newsletters) ? index.newsletters : [];
const podcasts = Array.isArray(index.podcasts) ? index.podcasts : [];
if (newsletters.length !== 349 || podcasts.length !== 289) {
  throw new Error(`Unexpected index counts: ${newsletters.length} newsletters, ${podcasts.length} podcasts.`);
}

const files = await listFiles(archiveRoot);
const manifestFiles = [];
let totalBytes = 0;
for (const filePath of files) {
  const fileStat = await stat(filePath);
  totalBytes += fileStat.size;
  manifestFiles.push({
    path: toPortablePath(path.relative(repositoryRoot, filePath)),
    bytes: fileStat.size,
    sha256: await sha256(filePath)
  });
}

const manifest = {
  schemaVersion: 1,
  generatedAt: new Date().toISOString(),
  archive: {
    root: 'archive',
    description: "Paid-subscriber Lenny's Newsletter and Podcast archive snapshot.",
    index: {
      path: 'archive/01-start-here/index.json',
      sha256: await sha256(indexPath),
      schemaVersion: index.schema_version,
      generatedAt: index.generated_at,
      newsletterCount: newsletters.length,
      podcastCount: podcasts.length
    },
    fileCount: manifestFiles.length,
    totalBytes,
    files: manifestFiles
  }
};

await writeFile(outputPath, `${JSON.stringify(manifest, null, 2)}\n`, 'utf8');
console.log(`Wrote ${path.relative(repositoryRoot, outputPath)} for ${manifestFiles.length} files (${totalBytes} bytes).`);
