export interface PageRecord {
  sourcePath: string;
  route: string;
  title: string;
}

export interface RouteRegistry {
  routeForWikiTarget(target: string): string | undefined;
}
