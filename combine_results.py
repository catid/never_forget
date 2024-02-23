import paramiko
import csv
import os

def download_csv_from_server(hostname, remote_path, local_path):
    """
    Downloads a CSV file from a server using SCP over SSH.
    """
    try:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(hostname)
        sftp = client.open_sftp()
        sftp.get(remote_path, local_path)
        sftp.close()
        client.close()
        print(f"Successfully downloaded {remote_path} from {hostname} to {local_path}")
    except Exception as e:
        print(f"Failed to download from {hostname}. Error: {e}")

def combine_csv_files(file_list, combined_file):
    """
    Combines multiple CSV files into one, assuming the same headers.
    """
    with open(combined_file, 'w', newline='') as fout:
        writer = csv.writer(fout)
        for i, file in enumerate(file_list):
            with open(file, 'r') as fin:
                reader = csv.reader(fin)
                if i == 0:
                    # Write headers from the first file
                    writer.writerow(next(reader))
                else:
                    # Skip headers for subsequent files
                    next(reader)
                # Write the rest of the data
                writer.writerows(reader)
            os.remove(file)  # Optionally remove the file after combining
        print(f"All files have been combined into {combined_file}")

def main():
    hosts_file = 'hosts.txt'
    remote_csv_path = '/home/catid/sources/never_forget/results.csv'  # Adjust the remote path
    local_directory = './temp_csvs'
    combined_csv_path = 'combined_results.csv'

    if not os.path.exists(local_directory):
        os.makedirs(local_directory)

    # Read hosts and credentials
    with open(hosts_file, 'r') as file:
        hosts = file.readlines()

    downloaded_files = []
    for host in hosts:
        hostname = host.strip().split(',')
        if not hostname:
            continue
        hostname = hostname[0]
        if hostname.startswith('#'):
            continue

        local_path = os.path.join(local_directory, f"{hostname}_results.csv")
        download_csv_from_server(hostname, remote_csv_path, local_path)
        downloaded_files.append(local_path)

    combine_csv_files(downloaded_files, combined_csv_path)

if __name__ == "__main__":
    main()
