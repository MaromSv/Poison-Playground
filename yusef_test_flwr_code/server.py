import flwr as fl

fl.server.start_server(
    server_address="0.0.0.0:8080", 
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=fl.server.strategy.FedAvg() # The FL model strategy can easily be changed here, flwr has a lot of options already
)