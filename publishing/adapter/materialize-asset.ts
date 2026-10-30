import {copyFile,link} from 'node:fs/promises';

// Optional local low-disk staging. Staged hardlinks must never be edited:
// they share source bytes. Astro reads them and emits independent static files.
// Default/CI builds still copy, and a failed hardlink is not silently copied.
export async function materializeAsset(source:string,destination:string,hardlink=false):Promise<void>{
  if(hardlink) await link(source,destination);
  else await copyFile(source,destination);
}
