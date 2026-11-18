
import re
import ipaddress
from urllib.parse import urlsplit


def extract_features(url: str) -> dict:
    """Extract 12 structural features from a URL."""

    url = str(url).strip()

    # urlsplit handles URLs with or without a scheme
    candidate = url if "://" in url else "//" + url

    try:
        parsed = urlsplit(candidate)
        hostname = parsed.hostname or ""
    except ValueError:
        parsed = None
        hostname = ""

    features = {
        # 1. Total URL length
        "url_length": len(url),

        # 2. Hostname length
        "domain_length": len(hostname),

        # 3. Number of dots
        "num_dots": url.count("."),

        # 4. Number of hyphens
        "num_hyphens": url.count("-"),

        # 5. Number of digits
        "num_digits": sum(char.isdigit() for char in url),

        # 6. Number of special characters
        "num_special_chars": len(
            re.findall(r"""[!@#$%^&*(),;'"{}\[\]|\\<>]""", url)
        ),

        # 7. Number of slashes
        "num_slashes": url.count("/"),

        # 8. Number of query parameters
        "num_params": (
            len(parsed.query.split("&"))
            if parsed and parsed.query
            else 0
        ),

        # 9. Number of subdomains (approximate)
        "num_subdomains": (
            max(len(hostname.split(".")) - 2, 0)
            if hostname
            else 0
        ),

        # 10. HTTPS present
        "has_https": int(url.lower().startswith("https://")),

        # 11. IPv4 or IPv6 address used as hostname
        "has_ip": 0,

        # 12. @ symbol present
        "has_at": int("@" in url),
    }

    try:
        ipaddress.ip_address(hostname)
        features["has_ip"] = 1
    except ValueError:
        pass

    return features


if __name__ == "__main__":
    sample_url = "https://secure-login.example.com/account?id=123"

    print("Sample URL:", sample_url)
    print("\nExtracted features:")

    for name, value in extract_features(sample_url).items():
        print(f"{name}: {value}")
