import matplotlib.pyplot as plt
import numpy as np
import sys
import os

STOP_THRESH = 2

class KMeans:
  def __init__(self, max_epochs, live_graph):
    self.K = 0
    self.max_epochs = max_epochs

    self.data = []
    self.cluster_pts = []
    self.prev_cluster_pts = []
    self.empty_clusters = []

    self.live_graph = live_graph
  # END __init__

  def add_data_pts(self, data):
    self.data = data
    self.N = len(data)

    # cluster assignment list per point, -1 = unassigned
    self.cluster_assignment = [-1] * self.N
  # END add_data_pts

  def add_cluster_pt(self, pt, color):
    #print("adding cluster pt", pt, color)

    self.cluster_pts.append([np.array(pt, dtype=np.float64), color])
    self.prev_cluster_pts.append([np.array([-9999, -9999], dtype=np.float64),
                                 color])
    self.K += 1
  # END add_cluster_pt

  def show_plot(self):
    plt.clf()

    # plot points
    colors = []
    for a in self.cluster_assignment:
      if (a != -1):
        colors.append(self.cluster_pts[a][1])
      else:
        colors.append("grey")

    plt.scatter(self.data[:,0], self.data[:,1], color=colors[:])
    
    # plot cluster_pts
    for c in self.cluster_pts:
      plt.scatter(c[0][0], c[0][1], color=c[1], marker="X", s=100)

    plt.show()
  # END show_plot

  def print_cluster_pts(self):
    print("printing cluster pts")
    for cpt in self.cluster_pts:
      print(cpt[1] + ": " + str(cpt[0]) + ",")
    print("\n")
  # END print_cluster_pts

  def show_plot_live(self):
    plt.clf()

    # plot points
    colors = []
    for a in self.cluster_assignment:
      if (a != -1):
        colors.append(self.cluster_pts[a][1])
      else:
        colors.append("grey")

    plt.scatter(self.data[:,0], self.data[:,1], color=colors[:])

    # plot cluster_pts
    for c in self.cluster_pts:
      plt.scatter(c[0][0], c[0][1], color=c[1], marker="X", s=100)

    plt.pause(0.01)
  # END show_plot_live

  def get_farthest_x_and_y(self, k):
    cp = self.cluster_pts[k]

    max_x = 0
    max_y = 0

    for p in self.get_pts_in_cluster(k):
      if (abs(p[0] - cp[0][0]) > max_x):
        max_x = abs(p[0] - cp[0][0])
      if (abs(p[1] - cp[0][1]) > max_y):
        max_y = abs(p[1] - cp[0][1])

    return (max_x, max_y)
  # END get_farthest_x_and_y

  def get_pts_in_cluster(self, k):
    pts = []
    for c, p in zip(self.cluster_assignment, self.data):
      if (c == k):
        pts.append(p)
    #print("pts in cluster", pts)
    return pts
  # END get_pts_in_cluster

  def dist(self, a, b):
    #print("getting dist between", a, b)
    return np.linalg.norm(a - b, axis=0)
  # END dist

  def get_closest_cluster(self, pt):
    dist_list = []
    for c in self.cluster_pts:
      dist_list.append(self.dist(pt, c[0]))
    #print("dist list", dist_list)
    d, i = min((d, i) for (i, d) in enumerate(dist_list)) 

    return i
  # END get_closest_cluster

  def is_conv(self):
    res = [False] * self.K

    for k in range(self.K):
      if (k in self.empty_clusters):
        res[k] = True
      elif (self.dist(self.cluster_pts[k][0], \
                      self.prev_cluster_pts[k][0]) <= STOP_THRESH):
        res[k] = True

    #print("is_conv", res)

    # return true if all are true
    return all(r for r in res)
  # END is_conv

  def assign_clusters(self):
    for i in range(self.N):
      self.cluster_assignment[i] = self.get_closest_cluster(self.data[i])
  # END assign_clusters

  def run_alg(self):
    if (self.live_graph):
      plt.ion()

    #for i in range(self.max_epochs):
    while (not self.is_conv()):
      self.assign_clusters()
      #self.print_cluster_pts()

      if (self.live_graph):
        self.show_plot_live()

      for k in range(self.K):
        # ignore empty clusters
        pts_in_cluster = self.get_pts_in_cluster(k)

        if (len(pts_in_cluster) == 0):
          self.empty_clusters.append(k)
          continue

        self.prev_cluster_pts[k][0] = self.cluster_pts[k][0]
        self.cluster_pts[k][0] = np.mean(pts_in_cluster, axis=0)
  # END run_alg
# END Kmeans

def main():
  fname = sys.argv[1]
  max_epoch = int(sys.argv[2])

  with open(fname, 'r') as f:
    lines = f.readlines()

  x_list = [float(l.split(' ')[0]) for l in lines]
  y_list = [float(l.split(' ')[1]) for l in lines]

  pts = []
  for i in range(len(x_list)):
    pts.append([x_list[i], y_list[i]])

  pts = np.array(pts)
  # print pts

  np.set_printoptions(threshold=np.inf)

  plt.rcParams["figure.figsize"] = [12, 10]

  km = KMeans(max_epoch, True)
  km.add_data_pts(pts)
  km.add_cluster_pt([0.4, 0.4], "red")
  km.add_cluster_pt([0.7, 0.6], "green")
  km.run_alg()

  km = KMeans(max_epoch, True)
  km.add_data_pts(pts)
  km.add_cluster_pt([0.4, 0.4], "red")
  km.add_cluster_pt([0.7, 0.6], "green")
  km.add_cluster_pt([0.4, 0.6], "blue")
  km.run_alg()

  km = KMeans(max_epoch, True)
  km.add_data_pts(pts)
  km.add_cluster_pt([0.4, 0.4], "red")
  km.add_cluster_pt([0.7, 0.6], "green")
  km.add_cluster_pt([0.4, 0.6], "blue")
  km.add_cluster_pt([0.7, 0.8], "pink")
  km.run_alg()

  km = KMeans(max_epoch, True)
  km.add_data_pts(pts)
  km.add_cluster_pt([0.4, 0.4], "red")
  km.add_cluster_pt([0.7, 0.6], "green")
  km.add_cluster_pt([0.4, 0.6], "blue")
  km.add_cluster_pt([0.7, 0.8], "pink")
  km.add_cluster_pt([0.5, 0.5], "orange")
  km.run_alg()

  km = KMeans(max_epoch, True)
  km.add_data_pts(pts)
  km.add_cluster_pt([0.4, 0.4], "red")
  km.add_cluster_pt([0.7, 0.6], "green")
  km.add_cluster_pt([0.4, 0.6], "blue")
  km.add_cluster_pt([0.7, 0.8], "pink")
  km.add_cluster_pt([0.5, 0.5], "orange")
  km.add_cluster_pt([0.3, 0.5], "cyan")
  km.run_alg()

if __name__ == "__main__":
  main()
