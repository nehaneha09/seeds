# Ultralytics YOLO 🚀, AGPL-3.0 license

import concurrent.futures
import statistics
import time
from typing import List, Tuple, Optional

import requests


class GCPRegions:
    def __init__(self):
        """Initializes the GCPRegions class with predefined Google Cloud Platform regions and their details."""
        self.regions = {
            "asia-east1": (1, "Taiwan", "China"),
            "asia-east2": (2, "Hong Kong", "China"),
            "asia-northeast1": (1, "Tokyo", "Japan"),
            "asia-northeast2": (1, "Osaka", "Japan"),
            "asia-northeast3": (2, "Seoul", "South Korea"),
            "asia-south1": (2, "Mumbai", "India"),
            "asia-south2": (2, "Delhi", "India"),
            "asia-southeast1": (2, "Jurong West", "Singapore"),
            "asia-southeast2": (2, "Jakarta", "Indonesia"),
            "australia-southeast1": (2, "Sydney", "Australia"),
            "australia-southeast2": (2, "Melbourne", "Australia"),
            "europe-central2": (2, "Warsaw", "Poland"),
            "europe-north1": (1, "Hamina", "Finland"),
            "europe-southwest1": (1, "Madrid", "Spain"),
            "europe-west1": (1, "St. Ghislain", "Belgium"),
            "europe-west10": (2, "Berlin", "Germany"),
            "europe-west12": (2, "Turin", "Italy"),
            "europe-west2": (2, "London", "United Kingdom"),
            "europe-west3": (2, "Frankfurt", "Germany"),
            "europe-west4": (1, "Eemshaven", "Netherlands"),
            "europe-west6": (2, "Zurich", "Switzerland"),
            "europe-west8": (1, "Milan", "Italy"),
            "europe-west9": (1, "Paris", "France"),
            "me-central1": (2, "Doha", "Qatar"),
            "me-west1": (1, "Tel Aviv", "Israel"),
            "northamerica-northeast1": (2, "Montreal", "Canada"),
            "northamerica-northeast2": (2, "Toronto", "Canada"),
            "southamerica-east1": (2, "São Paulo", "Brazil"),
            "southamerica-west1": (2, "Santiago", "Chile"),
            "us-central1": (1, "Iowa", "United States"),
            "us-east1": (1, "South Carolina", "United States"),
            "us-east4": (1, "Northern Virginia", "United States"),
            "us-east5": (1, "Columbus", "United States"),
            "us-south1": (1, "Dallas", "United States"),
            "us-west1": (1, "Oregon", "United States"),
            "us-west2": (2, "Los Angeles", "United States"),
            "us-west3": (2, "Salt Lake City", "United States"),
            "us-west4": (2, "Las Vegas", "United States"),
        }

    def tier1(self) -> List[str]:
        """Returns a list of GCP regions classified as tier 1 based on predefined criteria."""
        return [region for region, info in self.regions.items() if info[0] == 1]

    def tier2(self) -> List[str]:
        """Returns a list of GCP regions classified as tier 2 based on predefined criteria."""
        return [region for region, info in self.regions.items() if info[0] == 2]

    @staticmethod
    def _ping_region(region: str, retries: int = 1) -> Tuple[str, float, float, float, float]:
        """Pings a specified GCP region and returns latency statistics: mean, min, max, and standard deviation."""
        url = f"https://{region}-docker.pkg.dev"
        latencies = []
        for _ in range(retries):
            try:
                start_time = time.time()
                _ = requests.head(url, timeout=5)
                latency = (time.time() - start_time) * 1000  # convert latency to milliseconds
                if latency != float("inf"):
                    latencies.append(latency)
            except requests.RequestException:
                pass
        if not latencies:
            return region, float("inf"), float("inf"), float("inf"), float("inf")

        std_dev = statistics.stdev(latencies) if len(latencies) > 1 else 0
        return region, statistics.mean(latencies), std_dev, min(latencies), max(latencies)

    def lowest_latency(
        self,
        top: int = 1,
        verbose: bool = False,
        tier: Optional[int] = None,
        retries: int = 1,
    ) -> List[Tuple[str, float, float, float, float]]:
        if verbose:
            print(f"Testing GCP regions for latency (with {retries} {'retry' if retries == 1 else 'retries'})...")

        regions_to_test = [k for k, v in self.regions.items() if v[0] == tier] if tier else list(self.regions.keys())
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            results = list(executor.map(lambda r: self._ping_region(r, retries), regions_to_test))

        sorted_results = sorted(results, key=lambda x: x[1])

        if verbose:
            print(f"{'Region':<20} {'Location':<35} {'Tier':<5} {'Latency (ms)'}")
            for region, mean, std, min_, max_ in sorted_results:
                tier, city, country = self.regions[region]
                location = f"{city}, {country}"
                if mean == float("inf"):
                    print(f"{region:<20} {location:<35} {tier:<5} {'Timeout'}")
                else:
                    print(f"{region:<20} {location:<35} {tier:<5} {mean:.0f} ± {std:.0f} ({min_:.0f} - {max_:.0f})")
            print(f"\nLowest latency region{'s' if top > 1 else ''}:")
            for region, mean, std, min_, max_ in sorted_results[:top]:
                tier, city, country = self.regions[region]
                location = f"{city}, {country}"
                print(f"{region} ({location}, {mean:.0f} ± {std:.0f} ms ({min_:.0f} - {max_:.0f}))")

        return sorted_results[:top]


# Usage example
if __name__ == "__main__":
    regions = GCPRegions()
    top_3_latency_tier1 = regions.lowest_latency(top=3, verbose=True, tier=1, retries=3)
