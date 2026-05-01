"""
=============================================================
SIMULATION MCP (Model Context Protocol)
=============================================================

Ce module simule une architecture MCP appliquee a notre projet.

Rappel : le MCP (Model Context Protocol) d'Anthropic definit
3 composants :
1. MCP Host   : l'environnement principal (notre dashboard)
2. MCP Client : le composant qui envoie les requetes (notre LLMEnricher)
3. MCP Server : expose des outils/donnees de maniere isolee

Dans notre projet :
- MCP Host   = Dashboard Streamlit (app.py)
- MCP Client = LLMEnricher (enrichment.py)
- MCP Server = Chaque agent de scraping + le serveur de donnees

Ce module ajoute :
- Journalisation de toutes les requetes (audit trail)
- Controle des permissions par agent
- Validation des donnees avant transmission
"""

import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


# ==============================================================
# MCP SERVER : Serveur de donnees produits
# ==============================================================
class MCPServer:
    """
    Serveur MCP qui expose les donnees produits de maniere controlee.

    Principe d'isolation : le serveur ne donne acces qu'aux donnees
    autorisees selon les permissions du client.

    Permissions disponibles :
    - "read:products" : lire les donnees produits
    - "read:stats" : lire les statistiques
    - "read:top_k" : lire le classement Top-K
    - "write:reports" : generer des rapports
    """

    def __init__(self, server_name: str, data_path: str = None):
        self.server_name = server_name
        self.data_path = data_path
        self.permissions = set()
        self.request_log = []
        self.is_active = True
        logger.info(f"MCP Server '{server_name}' initialise")

    def add_permission(self, permission: str):
        """Ajoute une permission au serveur."""
        self.permissions.add(permission)
        logger.info(f"  Permission ajoutee : {permission}")

    def has_permission(self, permission: str) -> bool:
        """Verifie si une permission est accordee."""
        return permission in self.permissions

    def log_request(self, client_id: str, action: str, details: str = ""):
        """Journalise chaque requete (audit trail)."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "server": self.server_name,
            "client": client_id,
            "action": action,
            "details": details,
            "status": "pending"
        }
        self.request_log.append(entry)
        logger.info(f"  [LOG] {client_id} -> {self.server_name} : {action}")

    def handle_request(self, client_id: str, action: str, permission: str,
                       params: Dict = None) -> Dict:
        """
        Traite une requete d'un MCP Client.

        Verifie les permissions avant de repondre.
        Journalise toutes les requetes.
        """
        # Journalisation
        self.log_request(client_id, action, json.dumps(params or {}))

        # Verification des permissions
        if not self.has_permission(permission):
            self.request_log[-1]["status"] = "denied"
            logger.warning(f"  [REFUSE] {client_id} n'a pas la permission '{permission}'")
            return {
                "status": "error",
                "message": f"Permission '{permission}' non accordee",
                "server": self.server_name
            }

        # Traitement selon l'action
        self.request_log[-1]["status"] = "success"

        if action == "get_products":
            return self._get_products(params)
        elif action == "get_stats":
            return self._get_stats()
        elif action == "get_top_k":
            return self._get_top_k(params)
        else:
            return {"status": "error", "message": f"Action inconnue : {action}"}

    def _get_products(self, params):
        """Retourne les produits (avec filtres optionnels)."""
        import pandas as pd
        if not self.data_path:
            return {"status": "error", "message": "Pas de chemin de donnees configure"}

        df = pd.read_csv(self.data_path)
        if params and "store" in params:
            df = df[df['store_name'] == params['store']]
        if params and "max_price" in params:
            df = df[df['price'] <= params['max_price']]

        return {
            "status": "success",
            "count": len(df),
            "data": df.head(10).to_dict('records')
        }

    def _get_stats(self):
        """Retourne les statistiques globales."""
        import pandas as pd
        if not self.data_path:
            return {"status": "error", "message": "Pas de chemin de donnees configure"}

        df = pd.read_csv(self.data_path)
        return {
            "status": "success",
            "total_products": len(df),
            "stores": df['store_name'].unique().tolist(),
            "avg_price": round(df['price'].mean(), 2),
            "success_rate": round(df['produit_succes'].mean() * 100, 1)
        }

    def _get_top_k(self, params):
        """Retourne les Top-K produits."""
        import pandas as pd
        if not self.data_path:
            return {"status": "error", "message": "Pas de chemin de donnees configure"}

        df = pd.read_csv(self.data_path)
        k = params.get("k", 10) if params else 10
        topk = df.nlargest(k, 'final_score')

        return {
            "status": "success",
            "k": k,
            "data": topk[['title', 'price', 'store_name', 'final_score']].to_dict('records')
        }

    def get_audit_log(self) -> List[Dict]:
        """Retourne le journal des requetes (audit trail)."""
        return self.request_log


# ==============================================================
# MCP CLIENT : Client qui envoie les requetes
# ==============================================================
class MCPClient:
    """
    Client MCP qui envoie des requetes aux serveurs.

    Chaque client a un identifiant et connait les serveurs
    auxquels il peut se connecter.
    """

    def __init__(self, client_id: str):
        self.client_id = client_id
        self.connected_servers = {}
        logger.info(f"MCP Client '{client_id}' initialise")

    def connect_server(self, server: MCPServer):
        """Connecte le client a un serveur MCP."""
        self.connected_servers[server.server_name] = server
        logger.info(f"  Client '{self.client_id}' connecte au serveur '{server.server_name}'")

    def request(self, server_name: str, action: str, permission: str,
                params: Dict = None) -> Dict:
        """Envoie une requete a un serveur."""
        if server_name not in self.connected_servers:
            return {"status": "error", "message": f"Serveur '{server_name}' non connecte"}

        server = self.connected_servers[server_name]
        return server.handle_request(self.client_id, action, permission, params)


# ==============================================================
# MCP HOST : Environnement principal
# ==============================================================
class MCPHost:
    """
    MCP Host = l'environnement principal qui coordonne tout.

    Dans notre projet, le MCP Host est le Dashboard Streamlit.
    Il cree les serveurs et les clients, et orchestre les interactions.

    Responsabilites :
    - Initialiser les MCP Servers (un par source de donnees)
    - Creer les MCP Clients (un par module : LLM, Dashboard, etc.)
    - Gerer les permissions globales
    - Fournir l'audit trail complet
    """

    def __init__(self, host_name: str = "SmartCommerce Dashboard"):
        self.host_name = host_name
        self.servers = {}
        self.clients = {}
        logger.info(f"MCP Host '{host_name}' initialise")

    def create_server(self, name: str, data_path: str = None,
                      permissions: List[str] = None) -> MCPServer:
        """Cree et configure un MCP Server."""
        server = MCPServer(name, data_path)
        if permissions:
            for perm in permissions:
                server.add_permission(perm)
        self.servers[name] = server
        return server

    def create_client(self, client_id: str, connect_to: List[str] = None) -> MCPClient:
        """Cree un MCP Client et le connecte aux serveurs specifies."""
        client = MCPClient(client_id)
        if connect_to:
            for server_name in connect_to:
                if server_name in self.servers:
                    client.connect_server(self.servers[server_name])
        self.clients[client_id] = client
        return client

    def get_full_audit_log(self) -> List[Dict]:
        """Retourne le journal complet de tous les serveurs."""
        full_log = []
        for server in self.servers.values():
            full_log.extend(server.get_audit_log())
        full_log.sort(key=lambda x: x['timestamp'])
        return full_log

    def print_status(self):
        """Affiche l'etat complet de l'architecture MCP."""
        print(f"\n{'='*60}")
        print(f"  MCP HOST : {self.host_name}")
        print(f"{'='*60}")
        print(f"\n  Serveurs MCP ({len(self.servers)}) :")
        for name, server in self.servers.items():
            print(f"    - {name} | Permissions: {server.permissions} | Actif: {server.is_active}")
        print(f"\n  Clients MCP ({len(self.clients)}) :")
        for cid, client in self.clients.items():
            servers = list(client.connected_servers.keys())
            print(f"    - {cid} | Connecte a: {servers}")
        print(f"\n  Journal d'audit : {len(self.get_full_audit_log())} requetes")
        print(f"{'='*60}")


# ==============================================================
# DEMONSTRATION
# ==============================================================
def demo_mcp():
    """
    Demonstration de l'architecture MCP appliquee au projet.

    Montre comment les 3 composants (Host, Client, Server)
    interagissent de maniere responsable et securisee.
    """
    print("\n" + "=" * 60)
    print("  DEMONSTRATION MCP")
    print("  Architecture responsable pour Smart eCommerce")
    print("=" * 60)

    # 1. Creer le MCP Host (Dashboard)
    host = MCPHost("SmartCommerce Dashboard")

    # 2. Creer les MCP Servers
    shopify_server = host.create_server(
        name="Shopify Data Server",
        data_path="data/processed/products_clean.csv",
        permissions=["read:products", "read:stats", "read:top_k"]
    )

    analytics_server = host.create_server(
        name="Analytics Server",
        data_path="data/processed/products_clean.csv",
        permissions=["read:stats", "read:top_k", "write:reports"]
    )

    # 3. Creer les MCP Clients
    llm_client = host.create_client(
        client_id="LLM Enricher (Groq)",
        connect_to=["Shopify Data Server", "Analytics Server"]
    )

    dashboard_client = host.create_client(
        client_id="Dashboard Streamlit",
        connect_to=["Shopify Data Server", "Analytics Server"]
    )

    # 4. Afficher l'etat
    host.print_status()

    # 5. Simuler des requetes
    print("\n  --- Simulation de requetes ---\n")

    # Requete legale : le LLM demande les stats
    result = llm_client.request("Analytics Server", "get_stats", "read:stats")
    print(f"  LLM -> Analytics (get_stats) : {result['status']} | Produits: {result.get('total_products', 'N/A')}")

    # Requete legale : le dashboard demande les Top-5
    result = dashboard_client.request("Analytics Server", "get_top_k", "read:top_k", {"k": 5})
    print(f"  Dashboard -> Analytics (top_5) : {result['status']} | Top-K: {result.get('k', 'N/A')}")

    # Requete refusee : le LLM essaie d'ecrire (pas la permission)
    result = llm_client.request("Shopify Data Server", "write_data", "write:products")
    print(f"  LLM -> Shopify (write) : {result['status']} | {result.get('message', '')}")

    # 6. Afficher le journal d'audit
    print(f"\n  --- Journal d'audit ---\n")
    for entry in host.get_full_audit_log():
        print(f"  [{entry['status']:>7}] {entry['timestamp'][:19]} | "
              f"{entry['client']} -> {entry['server']} : {entry['action']}")

    print(f"\n{'='*60}")
    print("  Fin de la demonstration MCP")
    print(f"{'='*60}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    demo_mcp()
