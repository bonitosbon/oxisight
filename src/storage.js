import { openDB } from 'idb';
const database = () => openDB('oxisight',1,{upgrade(db){db.createObjectStore('scans',{keyPath:'id'});}});
export async function getScans(){return (await (await database()).getAll('scans')).sort((a,b)=>b.createdAt-a.createdAt);}
export async function saveScan(scan){await (await database()).put('scans',scan);}
export async function removeScan(id){await (await database()).delete('scans',id);}
